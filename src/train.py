import argparse

from ultralytics import YOLO
from utils import freeze_layer


if __name__ == "__main__":

    # parse command line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate the YOLO model for cassowary detection.")
    parser.add_argument("--action", type=str, default="train", help='Action to perform: "train" or "eval"')
    parser.add_argument("--batch_size", type=int, default=96, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=2, help="Num of epochs to train")
    parser.add_argument(
        "--eval_model_path",
        type=str,
        default="/app/data/training/stage3/weights/best.pt",
        help="Path to eval model weights",
    )
    parser.add_argument("--stage", type=int, default=1, help="Stage of training to execute")
    args = parser.parse_args()

    # params
    action = args.action.lower()
    stage = args.stage
    project_name = "stage" + str(stage)
    pre_path = "/app/"

    if action == "train":
        # identify correct weights
        model_path = "yolov8m.pt"
        if stage == 2:
            model_path = pre_path + "data/training/stage1/weights/best.pt"
        elif stage == 3:
            model_path = pre_path + "data/training/stage2/weights/best.pt"

        # train
        model = YOLO(model_path)
        model.add_callback("on_train_start", freeze_layer)
        model.train(
            data=pre_path + "trainer/data.yaml",
            project=pre_path + "data/training",
            name=project_name,
            epochs=args.epochs,
            imgsz=640,
            pretrained=True,
            batch=args.batch_size,
            workers=0,
        )

    if action == "eval":
        if stage == 1:
            eval_model_path = pre_path + "data/training/stage1/weights/best.pt"
        elif stage == 2:
            eval_model_path = pre_path + "data/training/stage2/weights/best.pt"         
        model = YOLO(eval_model_path)
        results = model.val(data= pre_path + "trainer/data.yaml", workers=0) 
                            
