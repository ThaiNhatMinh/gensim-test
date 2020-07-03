
from Model import LDAModel, LSIModel
import shutil

text1="""Photoshop layers are like sheets of stacked acetate. You can see through transparent areas of a layer to the layers below.
You move a layer to position the content on the layer, like sliding a sheet of acetate in a stack. You can also change the opacity of a layer to make content partially transparent."""

text2="""You use layers to perform tasks such as compositing multiple images, adding text to an image, or adding vector graphic shapes.
You can apply a layer style to add a special effect such as a drop shadow or a glow."""

text3="""A new image has a single layer. The number of additional layers, layer effects, and layer sets you can add to an image is limited only by your computer’s memory.
You work with layers in the Layers panel. Layer groups help you organize and manage layers. You can use groups to arrange your layers in a logical order and to reduce clutter in the Layers panel. You can nest groups within other groups. You can also use groups to apply attributes and masks to multiple layers simultaneously.
For some great tips for working with layers, see the tutorial video Organize with layers and layer groups."""

text4="""You can use video layers to add video to an image. After importing a video clip into an image as a video layer, you can mask the layer, transform it, apply layer effects, paint on individual frames, or rasterize an individual frame and convert it to a standard layer.
Use the Timeline panel to play the video within the image or to access individual frames."""

documents1 =  ([text1, text4, text2, text3], "layer in photoshop")

documents2 = (["""Adding texture to photographs was happening long before the invention of Photoshop and other editing programs. In the days of the darkroom, we would scratch negatives with pins, sand paper them, stain the photographic paper with fixer before exposing, layer two negatives on top of each other in the enlarger, or push our film ISO to increase grain. These days, with editing programs being our digital darkroom, we simply have yet another way of adding texture to photographs.""",
              """In digital photography terms it’s simply another layer added to your photograph in an editing program, usually an image of some sort of textural surface, such as paper, wood, concrete, etc., but anything at all can be a texture. They can be photographed, scanned or even made in Photoshop.""",
              """With the right texture overlay, and application of it,  you can add an extra level of depth and feeling to your photograph. You can use them for anything – from adding a vintage or grunge look to your photographs, to creating fine art pieces.
One of the best uses is to rescue a photograph that just isn’t quite working. I’ve been told no texture overlay will save a terrible photograph. While this is true for the most part, sometimes it can transform an otherwise unusable image to something more promising.
Textures can be added to almost any kind of image. If you’ve ever downloaded a photography app for your Smartphone, you have most likely had them add a texture with the app’s built-in filters.""",
              """You don’t need to create your own texture to get started. There are many pre-made, free textures available on the internet. A quick Google search will bring up a bunch of free texture sites. But not all textures are created equally, or usable legally. You need to look for textures that are a decent size and resolution, a 200px/72dpi texture over a 3200 px/300dpi image probably isn’t going to work so well. You’ll also want to make sure the texture has the right copyright permissions. Sites like deviantart.com have many stock textures offered free by their artists for personal use. These artists ask that you simply return the favor by sending them a link to the image you created using their work. Other sites like freetstocktextures.com offer their images copyright free for personal and commercial use, as long as you aren’t reselling the texture images themselves."""],
"texture in photoshop")
documents3 = ([
    "The Brush tool allows you to paint on any layer, much like a real paintbrush. You'll also have different settings to choose from, which can help you customize it for different situations. Once you know how to use the Brush tool, you'll notice that many other tools, including the Eraser and the Spot Healing Brush, use a similar group of settings.",
    "It's easy to use the Brush tool to paint in your document. Simply locate and select the Brush tool from the Tools panel, then click and drag in the document window to paint. You can also press the B key on your keyboard to select the Brush tool at any time.",
    "To choose a different brush color, click the top-most color in the Color Picker Tool (this is known as the Foreground Color), then select the desired color from the dialog box.",
    "You'll also be able to customize different settings for the Brush tool from the Control panel near the top of the screen.",
], "brush in photoshop")

def test(document, model=LSIModel):
    print(document[1])
    for num_topic in range(1, 7, 1):
        print("Num topic: {}".format(num_topic))
        lda = model(document[0], num_topics=num_topic)
        lda_sim = lda.similarity(document[1])
        # print(sim[0])
        for i, sim in lda_sim:
            print((i, sim))
        shutil.rmtree("models")

test(documents3, LDAModel)