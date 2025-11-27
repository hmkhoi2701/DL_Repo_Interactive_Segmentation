import glob
import json
from time import time
from demo.demo_utils import load_rgb, load_bin_mask, dice_iou, best_rater, neutral_binary,\
    suggest_next_click, overlay, valid_raters

import cfg
import torch
import os
from utils import get_network
import numpy as np
from PIL import Image
import torch.nn.functional as F

from models.sam.modeling import EMWeights, EMMeanVariance

args = cfg.parse_args()
args.image_size = 128
args.sam_ckpt = './checkpoint/medsam_vit_b.pth'
args.weights = './checkpoint/checkpoint_aespa_lidc.pth'

GPUdevice = torch.device('cuda', args.gpu_device)
net = get_network(args, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
net.EM_weights = EMWeights(n_components=args.n_components).to(GPUdevice)
net.EM_mean_variance = EMMeanVariance(se_dim = 256, pe_dim = 256, n_components=args.n_components).to(GPUdevice)

checkpoint_file = os.path.join(args.weights)
checkpoint = torch.load(checkpoint_file, weights_only = False)

state_dict = checkpoint['state_dict']
net.load_state_dict(state_dict)
net.EM_weights.weights = checkpoint['EM_weights']
net.EM_mean_variance.means = checkpoint['EM_means']
net.EM_mean_variance.variances = checkpoint['EM_variances']

print(f'=> resuming from {args.weights}, model load completed!')

# CONSTANTS
IMAGE_PATH = './demo/image.png'
RATER_GLOB = os.path.join('./demo/', "rater_*.png")

class AESPAInferencer:
    def __init__(self, net: torch.nn.Module = net, device: str="cuda", image_size: int=args.image_size, multimask_output: bool=False):
        self.net  = net.eval().to(device)
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.multimask  = bool(multimask_output)
        self.orig_hw = None
        self.img_tensor = None
        self.scale_y = None
        self.scale_x = None
        self.coords = []
        self.last_pred = None
        self.pred_masks_weights_list = None  # (1,K)

    def _preprocess_image(self, image_rgb):
        H,W,_ = image_rgb.shape
        self.orig_hw = (H,W)
        self.scale_y = self.image_size / float(H)
        self.scale_x = self.image_size / float(W)
        im = Image.fromarray(image_rgb).resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(im).astype(np.float32) / 255.0
        arr = arr.transpose(2,0,1)  # CHW
        return torch.from_numpy(arr).unsqueeze(0).to(self.device)

    def set_image(self, image_rgb):
        self.img_tensor = self._preprocess_image(image_rgb)
        self.coords.clear()
        self.last_pred = None
        with torch.no_grad():
            w0 = torch.tensor(self.net.EM_weights.weights, dtype=torch.float32, device=self.device)
        self.pred_masks_weights_list = w0.unsqueeze(0)  # (1,K)

    def add_click(self, y: int, x: int, label: int):
        self.coords.append((int(y), int(x), int(label)))

    def undo(self):
        if self.coords: self.coords.pop()

    def clear(self):
        self.coords.clear()
        self.last_pred = None
        with torch.no_grad():
            w0 = torch.tensor(self.net.EM_weights.weights, dtype=torch.float32, device=self.device)
        self.pred_masks_weights_list = w0.unsqueeze(0)

    @torch.no_grad()
    def predict(self):
        assert self.img_tensor is not None, "Call set_image(image) first."
        # Build prompt tensors (1, N, 2) and (1, N)
        if len(self.coords) > 0:
            pts = []; labs = []
            for (y,x,l) in self.coords:
                pts.append([y * self.scale_y, x * self.scale_x])  # (y, x)
                labs.append(int(l))
            coords_t = torch.tensor(pts, dtype=torch.float32, device=self.device).unsqueeze(0)
            labels_t = torch.tensor(labs, dtype=torch.int64, device=self.device).unsqueeze(0)
        else:
            coords_t = torch.empty((1,0,2), dtype=torch.float32, device=self.device)
            labels_t = torch.empty((1,0),   dtype=torch.int64,   device=self.device)
        
        se, de = self.net.prompt_encoder(points=(coords_t, labels_t), boxes=None, masks=None)
        pe = self.net.prompt_encoder.get_dense_pe().to(self.device)

        means, variances = self.net.EM_mean_variance(se, pe)

        weights_vec = self.net.EM_weights(self.pred_masks_weights_list)  # (1,K)
        weights_vec = (weights_vec.mean(dim=0) / weights_vec.sum()).to(self.device)  # (K,)

        imge = self.net.image_encoder(self.img_tensor, weights_vec, means, variances, num_sample=1)
        if isinstance(imge, (list, tuple)):
            imge = imge[0]
            
        if isinstance(imge, torch.Tensor):
            if imge.dim() == 5:                       # [S,B,C,H,W]
                if imge.size(0) == 1:
                    imge = imge.squeeze(0)            # -> [B,C,H,W]
                else:
                    imge = imge.mean(dim=0)

        pred_logits, _ = self.net.mask_decoder(
            image_embeddings=imge,
            image_pe=pe,
            sparse_prompt_embeddings=se,
            dense_prompt_embeddings=de,
            multimask_output=self.multimask,
        )

        pred_imgsz = F.interpolate(pred_logits, size=(self.image_size, self.image_size),
                                   mode="bilinear", align_corners=False)
        prob_imgsz = torch.sigmoid(pred_imgsz)[0,0].detach().cpu().numpy()

        # Update EM weights list for the next click
        flat = torch.from_numpy(prob_imgsz.reshape(-1)).to(self.device, dtype=torch.float32)
        next_w = self.net.EM_weights.compute_weights(flat, weights_vec, means, variances)
        self.pred_masks_weights_list = next_w.unsqueeze(0)

        # Back to original size
        prob = torch.from_numpy(prob_imgsz).unsqueeze(0).unsqueeze(0).to(self.device)
        prob = F.interpolate(prob, size=self.orig_hw, mode="bilinear", align_corners=False)[0,0].cpu().numpy()
        binm = (prob > 0.5).astype(np.float32)
        return prob, binm

def _load_fixed_data():
    if not os.path.isfile(IMAGE_PATH):
        raise FileNotFoundError(f"Missing {IMAGE_PATH}")
    image = load_rgb(IMAGE_PATH)
    H,W,_ = image.shape

    mask_paths = sorted(glob.glob(RATER_GLOB))
    if len(mask_paths) == 0:
        raise FileNotFoundError(f"No masks found: {RATER_GLOB}")
    raters = np.stack([load_bin_mask(p, (H,W)) for p in mask_paths], axis=0).astype(np.float32)
    names  = [os.path.basename(p) for p in mask_paths]
    return image, raters, names

def build_app():
    import gradio as gr

    VIEW_SCALE = 4

    def _up(img_np: np.ndarray) -> np.ndarray:
        return np.array(
            Image.fromarray(img_np).resize(
                (img_np.shape[1] * VIEW_SCALE, img_np.shape[0] * VIEW_SCALE),
                Image.NEAREST
            )
        )

    def _overlay_dual(image_rgb: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
        """
        Overlay: GT=green, Pred=red.
        """
        img = image_rgb.astype(np.float32).copy()
        H, W, _ = img.shape
        assert gt_mask.shape == (H, W) and pred_mask.shape == (H, W)
        alpha = 0.45

        gt3  = np.repeat((gt_mask > 0)[..., None], 3, axis=2).astype(np.float32)
        pr3  = np.repeat((pred_mask > 0)[..., None], 3, axis=2).astype(np.float32)

        green = np.zeros_like(img); green[..., 1] = 255
        red   = np.zeros_like(img); red[..., 0]   = 255

        img = img * (1 - alpha * gt3) + green * (alpha * gt3)
        img = img * (1 - alpha * pr3) + red   * (alpha * pr3)
        return img.clip(0, 255).astype(np.uint8)

    def _gallery_items(image, raters, names, pred_mask):
        """[(image, caption), ...] : overlay dual (GT vs Current) + Dice/IoU."""
        items = []
        for i in range(raters.shape[0]):
            d, j = dice_iou(pred_mask, raters[i])
            cap = f"{names[i]} | Dice {d:.3f} | IoU {j:.3f}"
            vis = _up(_overlay_dual(image, raters[i], pred_mask))
            items.append((vis, cap))
        return items

    def _follow_caption(s: dict) -> str:
        idx, _ = best_rater(s["pred"], s["raters"], s["valid"])
        if idx is None or idx < 0:
            return f"Following: -  |  Clicks: {len(s['clicks'])}"
        d, j = dice_iou(s["pred"], s["raters"][idx])
        return f"Following: {s['names'][idx]}  |  Dice {d:.3f} · IoU {j:.3f}  |  Clicks: {len(s['clicks'])}"

    def _btn_updates(s: dict):
        """Enable/disable cho Undo/Redo/Confirm."""
        undo_ok    = len(s["clicks"]) > 0
        redo_ok    = len(s["redo_stack"]) > 0
        confirm_ok = (s["pending"] is not None)
        return (gr.update(interactive=undo_ok),
                gr.update(interactive=redo_ok),
                gr.update(interactive=confirm_ok))

    def _post_update(s: dict):
        clicks_to_draw = [s["pending"]] if s["pending"] is not None else []
        main_vis = _up(overlay(s["image"], s["pred"], clicks_to_draw, None))
        gallery  = _gallery_items(s["image"], s["raters"], s["names"], s["pred"])
        cap      = _follow_caption(s)
        u_upd, r_upd, c_upd = _btn_updates(s)
        return main_vis, gallery, cap, u_upd, r_upd, c_upd

    # --- model wrapper ---
    infer = AESPAInferencer(net, device=GPUdevice, image_size=args.image_size, multimask_output=False)

    # --- load fixed files ---
    image, raters, names = _load_fixed_data()
    valid = valid_raters(raters)
    infer.set_image(image)

    # --- initial state ---
    pred = neutral_binary(raters, valid)
    state_init = {
        "image": image, "raters": raters, "names": names, "valid": valid,
        "pred": pred,
        "clicks": [],          # confirmed clicks
        "redo_stack": [],      # redo history
        "pending": None,       # (y,x,label) waiting for confirm
        "suggested_xy": None,
    }

    with gr.Blocks(title="AESPA Model Demo") as demo:
        gr.Markdown("### AESPA Model Demo")

        state = gr.State(state_init)

        with gr.Row():
            label_radio = gr.Radio(choices=["positive","negative"], value="positive", label="Click label")
            confirm_btn = gr.Button("Confirm click", interactive=False)
            suggest_btn = gr.Button("Suggest next click") 
            undo_btn    = gr.Button("Undo", interactive=False)
            redo_btn    = gr.Button("Redo", interactive=False)
            clear_btn   = gr.Button("Clear")

        with gr.Row():
            canvas = gr.Image(
                value=_up(overlay(image, pred, [], None)),
                label="Prediction (click to add a point)",
                type="numpy",
                interactive=True,
                sources=[],
                height=128 * VIEW_SCALE,
                width=128 * VIEW_SCALE
            )

            r_gallery = gr.Gallery(
                value=_gallery_items(image, raters, names, pred),
                label="Rater overlays: GT (green) vs Current (red)",
                columns=4,
                height=128 * VIEW_SCALE
            )

        follow_md = gr.Markdown(_follow_caption({
            "pred": pred, "raters": raters, "valid": valid, "names": names, "clicks": []
        }))

        # ----------------- callbacks -----------------

        def on_click(evt: gr.SelectData, lab, s):
            s = dict(s)
            if evt is None or evt.index is None:
                main_vis, gallery, cap, u_upd, r_upd, c_upd = _post_update(s)
                return s, main_vis, gallery, cap, u_upd, r_upd, c_upd

            # map UI -> 128x128
            x_ui, y_ui = int(evt.index[0]), int(evt.index[1])
            H, W = s["image"].shape[:2]
            x = int(round(x_ui / VIEW_SCALE)); y = int(round(y_ui / VIEW_SCALE))
            x = max(0, min(W - 1, x)); y = max(0, min(H - 1, y))
            lbl = 1 if lab == "positive" else 0

            s["pending"] = (y, x, lbl)

            main_vis, gallery, cap, u_upd, r_upd, c_upd = _post_update(s)
            return s, main_vis, gallery, cap, u_upd, r_upd, c_upd

        canvas.select(on_click, [label_radio, state],
                      [state, canvas, r_gallery, follow_md, undo_btn, redo_btn, confirm_btn])

        def on_confirm(s):
            s = dict(s)
            if s["pending"] is not None:
                y, x, lbl = s["pending"]
                s["clicks"].append((int(y), int(x), int(lbl)))
                s["redo_stack"].clear()
                s["pending"] = None

                infer.add_click(int(y), int(x), int(lbl))
                _, binm = infer.predict()
                s["pred"] = binm

            main_vis, gallery, cap, u_upd, r_upd, c_upd = _post_update(s)
            return s, main_vis, gallery, cap, u_upd, r_upd, c_upd

        confirm_btn.click(on_confirm, [state],
                          [state, canvas, r_gallery, follow_md, undo_btn, redo_btn, confirm_btn])

        def on_suggest(s):
            s = dict(s)
            ref_i, _ = best_rater(s["pred"], s["raters"], s["valid"])
            if ref_i is None or ref_i < 0:
                main_vis, gallery, cap, u_upd, r_upd, c_upd = _post_update(s)
                return s, main_vis, gallery, cap, u_upd, r_upd, c_upd

            sug = suggest_next_click(s["pred"], s["raters"][ref_i])
            if sug is not None:
                y, x = int(sug[0]), int(sug[1])
                lbl = int(s["raters"][ref_i, y, x])
                s["pending"] = (y, x, lbl)

            main_vis, gallery, cap, u_upd, r_upd, c_upd = _post_update(s)
            return s, main_vis, gallery, cap, u_upd, r_upd, c_upd

        suggest_btn.click(on_suggest, [state],
                          [state, canvas, r_gallery, follow_md, undo_btn, redo_btn, confirm_btn])

        def on_undo(s):
            s = dict(s)
            if s["clicks"]:
                s["redo_stack"].append(s["clicks"].pop(-1))
                infer.undo()
                if s["clicks"]:
                    _, binm = infer.predict()
                    s["pred"] = binm
                else:
                    infer.clear()
                    s["pred"] = neutral_binary(s["raters"], s["valid"])

            main_vis, gallery, cap, u_upd, r_upd, c_upd = _post_update(s)
            return s, main_vis, gallery, cap, u_upd, r_upd, c_upd

        undo_btn.click(on_undo, [state],
                       [state, canvas, r_gallery, follow_md, undo_btn, redo_btn, confirm_btn])

        def on_redo(s):
            s = dict(s)
            if s["redo_stack"]:
                y, x, lbl = s["redo_stack"].pop(-1)
                s["clicks"].append((int(y), int(x), int(lbl)))
                infer.add_click(int(y), int(x), int(lbl))
                _, binm = infer.predict()
                s["pred"] = binm

            main_vis, gallery, cap, u_upd, r_upd, c_upd = _post_update(s)
            return s, main_vis, gallery, cap, u_upd, r_upd, c_upd

        redo_btn.click(on_redo, [state],
                       [state, canvas, r_gallery, follow_md, undo_btn, redo_btn, confirm_btn])

        def on_clear(s):
            s = dict(s)
            s["clicks"].clear()
            s["redo_stack"].clear()
            s["pending"] = None
            infer.clear()
            s["pred"] = neutral_binary(s["raters"], s["valid"])
            main_vis, gallery, cap, u_upd, r_upd, c_upd = _post_update(s)
            return s, main_vis, gallery, cap, u_upd, r_upd, c_upd

        clear_btn.click(on_clear, [state],
                        [state, canvas, r_gallery, follow_md, undo_btn, redo_btn, confirm_btn])

    return demo




if __name__ == "__main__":
    demo = build_app()
    demo.launch(server_port=7860, server_name=None)