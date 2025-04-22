from ultralytics import YOLOE

# Prompt free scenario
# Here the model detects anything and everything that it sees
# Not guided by any prompts
model = YOLOE("yoloe-11l-seg-pf.pt")
results= model.predict(
    "path/to/prompt_free.mp4",  # Here, replace this with path to video of your choice
    show=True,
    save=True, # can save if wanted else set false  
    show_conf=False
)
for r in results:
    boxes= r.boxes.xyxy.cpu().tolist()
    cls= r.boxes.cls.cpu().tolist()
    for b,c in zip(boxes,cls):
        print(f"box: {b}, cls: {model.names[c]}")