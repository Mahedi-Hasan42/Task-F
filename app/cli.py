import argparse
from pathlib import Path

import inventory as inv

def pick_files():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    paths = filedialog.askopenfilenames(
        title="Image select koro",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")],
    )
    root.destroy()
    return list(paths)

def pick_class_manually(classes):
    for i, name in enumerate(classes):
        print(f"  {i:2d}. {name}")
    while True:
        s = input("Right class number (blank = skip): ").strip()
        if s == "":
            return None
        if s.isdigit() and int(s) < len(classes):
            return classes[int(s)]
        print("wrong number, again submit.")

def handle(clf, source, res, threshold):
    label, conf = res["label"], res["confidence"]
    print(f"\n[INPUT] {source}")
    print(f"[PREDICT] {label}  confidence={conf:.4f}  ({res['ms']:.0f} ms)")

    if conf < threshold:
        inv.log_scan(source, label, conf, "REJECTED")
        print(f"[REJECTED] confidence {conf:.4f} < threshold {threshold}. "
              f"Product unkown, stock no change.")
        return

    ans = input(f"Is this product '{label}'? [y = yes / n = no / c = can i select name]: ").strip().lower()
    if ans == "y":
        qty = inv.add_stock(label)
        inv.log_scan(source, label, conf, "CONFIRMED", label)
        print(f"[CONFIRMED] {label}: stock wright now {qty}")
    elif ans == "c":
        real = pick_class_manually(clf.classes)
        if real is None:
            inv.log_scan(source, label, conf, "USER_DENIED")
            print("[FAILED] User not input, stock no change.")
        else:
            qty = inv.add_stock(real)
            inv.log_scan(source, label, conf, "CORRECTED", real)
            print(f"[CORRECTED] model tell '{label}', Real '{real}': stock right now {qty}")
    else:
        inv.log_scan(source, label, conf, "USER_DENIED")
        print("[FAILED] User reply negative, stock no change.")

def capture_frame():
    import cv2
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera did not open.")
        return None
    frame = None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        cv2.imshow("SPACE = capture, Q = quit", f)
        k = cv2.waitKey(1) & 0xFF
        if k == 32:
            frame = f
            break
        if k in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()
    return frame

def cmd_scan(args):
    from predict import Classifier          
    clf = Classifier(threshold=args.threshold)
    print(f"Model ready. Threshold = {clf.threshold}")

    if args.camera:
        frame = capture_frame()
        if frame is None:
            return
        handle(clf, "camera", clf.predict_frame(frame), clf.threshold)
        return

    paths = pick_files() if args.pick else args.image
    if not paths:
        print("no image selected .")
        return
    for p in paths:
        if not Path(p).exists():
            print(f"[ERROR] No File: {p}")
            continue
        handle(clf, str(p), clf.predict_file(p), clf.threshold)


def cmd_stats(args):
    rows, n_products, total = inv.stats()
    print(f"Product: {n_products}   Net quantity: {total}")
    for name, q in rows:
        print(f"  {name:40s} {q}")


def cmd_logs(args):
    s = inv.log_summary()
    print(f"Net scan: {s['total']}")
    print(f"  Successful: {s['successful']}")
    print(f"  Failed:     {s['failed']}")
    for k, v in sorted(s["by_status"].items()):
        print(f"    {k}: {v}")

def main():
    inv.init_db()
    p = argparse.ArgumentParser(description="Inventory scanner")
    sub = p.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("scan", help="scan from image or camera")
    g = sc.add_mutually_exclusive_group(required=True)
    g.add_argument("--pick", action="store_true", help="Open file select window")
    g.add_argument("--image", nargs="+", help="image file path")
    g.add_argument("--camera", action="store_true", help="One frame from webcam")
    sc.add_argument("--threshold", type=float, default=None,
                    help="confidence threshold (default: artifacts/threshold.json)")
    sc.set_defaults(func=cmd_scan)

    sub.add_parser("stats", help="inventory statistics").set_defaults(func=cmd_stats)
    sub.add_parser("logs", help="scan log summary").set_defaults(func=cmd_logs)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()