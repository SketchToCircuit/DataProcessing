import Tools.PinDetection.pindetection as pd
import random

components = pd.import_components('./exported_data/data.json')

random_type = random.sample(components.keys(), k=1)[0]
cmp = random.sample(components[random_type], k=1)[0]

cmp = cmp.load()
cmp.scale(1)
cmp.rotate(180)

cmp.label_img.shape[::-1]
cmp.component_img[::-1]

pd.combine_images(cmp.component_img, cmp.label_img, cmp.label_offset)

[*cmp.pins.values()][0].direction