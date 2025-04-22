from ultralytics import YOLOE

# Now we guide the model using text prompts
model= YOLOE('yoloe-11l-seg.pt')
names= ['camel']        # Replace with whatever prompt you are interested in the video
model.set_classes(names, model.get_text_pe(names))
model.predict(
    'path/to/text_prompt.mp4',   # Replace with the path to your video of choice
    show=True,
    save=True,
    show_conf=True
)