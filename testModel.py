
from Model import LDAModel, LSIModel
import shutil


documents1 = (["Photoshop layers are like sheets of stacked acetate. You can see through transparent areas of a layer to the layers below. You move a layer to position the content on the layer, like sliding a sheet of acetate in a stack. You can also change the opacity of a layer to make content partially transparent.",
               "You can use video layers to add video to an image. After importing a video clip into an image as a video layer, you can mask the layer, transform it, apply layer effects, paint on individual frames, or rasterize an individual frame and convert it to a standard layer. Use the Timeline panel to play the video within the image or to access individual frames.",
               "You use layers to perform tasks such as compositing multiple images, adding text to an image, or adding vector graphic shapes. You can apply a layer style to add a special effect such as a drop shadow or a glow.",
               "A new image has a single layer. The number of additional layers, layer effects, and layer sets you can add to an image is limited only by your computer’s memory.You work with layers in the Layers panel. Layer groups help you organize and manage layers. You can use groups to arrange your layers in a logical order and to reduce clutter in the Layers panel. You can nest groups within other groups. You can also use groups to apply attributes and masks to multiple layers simultaneously.For some great tips for working with layers, see the tutorial video Organize with layers and layer groups.",
               "Sometimes layers don’t contain any apparent content. For example, an adjustment layer holds color or tonal adjustments that affect the layers below it. Rather than edit image pixels directly, you can edit an adjustment layer and leave the underlying pixels unchanged.A special type of layer, called a Smart Object, contains one or more layers of content. You can transform (scale, skew, or reshape) a Smart Object without directly editing image pixels. Or, you can edit the Smart Object as a separate image even after placing it in a Photoshop image. Smart Objects can also contain smart filter effects, which allow you to apply filters non-destructively to images so that you can later tweak or remove the filter effect. See Nondestructive editing and Work with Smart Objects.",
               ],
"layer in photoshop")

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

documents4 = ([
                    "Photoshop uses the foreground color to paint, fill, and stroke selections and the background color to make gradient fills and fill in the erased areas of an image. The foreground and background colors are also used by some special effects filters.",
                    "You can designate a new foreground or background color using the Eyedropper tool, the Color panel, the Swatches panel, or the Adobe Color Picker.",
                    "The default foreground color is black, and the default background color is white. (In an alpha channel, the default foreground is white, and the background is black.)",
                    "The current foreground color appears in the upper color selection box in the toolbox; the current background color appears in the lower box.",
                    "The Eyedropper tool samples color to designate a new foreground or background color. You can sample from the active image or from anywhere else on the screen. ",
                    "To circle the Eyedropper tool with a ring that previews the sampled color above the current foreground color, select Show Sampling Ring. (This option requires OpenGL. See Enable OpenGL and optimize GPU settings.)",
                    "To select a new foreground color, click in the image. Alternatively, position the pointer over the image, press the mouse button, and drag anywhere on the screen. The foreground color selection box changes dynamically as you drag. Release the mouse button to pick the new color.",
                    "To select a new background color, Alt-click (Windows) or Option-click (Mac OS) in the image. Alternatively, position the pointer over the image, press Alt (Windows) or Options (Mac OS), press the mouse button, and drag anywhere on the screen. The background color selection box changes dynamically as you drag. Release the mouse button to pick the new color.",
                    "To use the Eyedropper tool temporarily to select a foreground color while using any painting tool, hold down Alt (Windows) or Option (Mac OS).",
                    "In the Adobe Color Picker, you choose colors using four color models: HSB, RGB, Lab, and CMYK. Use the Adobe Color Picker to set the foreground color, background color, and text color. You can also set target colors for different tools, commands, and options.",
                    "You can configure the Adobe Color Picker to let you choose only colors that are part of the web-safe palette or choose from specific color systems. You can also access an HDR (high dynamic range) picker to choose colors for use in HDR images.",
                    "The Color field in the Adobe Color Picker displays color components in HSB color mode, RGB color mode, and Lab color mode. If you know the numeric value of the color you want, you can enter it into the text fields. You can also use the color slider and the color field to preview a color to choose. As you adjust the color using the color field and color slider, the numeric values are adjusted accordingly. The color box to the right of the color slider displays the adjusted color in the top section and the original color in the bottom section. Alerts appear if the color is not a web-safe color &nbsp; or is out of gamut &nbsp;for printing (non-printable)&nbsp;.",
                    "When you select a color in the Adobe Color Picker, it simultaneously displays the numeric values for HSB, RGB, Lab, CMYK, and hexadecimal numbers. This is useful for viewing how the different color models describe a color.",
                    "Although Photoshop uses the Adobe Color Picker by default, you can use a different color picker than the Adobe Color Picker by setting a preference. For example, you can use the built-in color picker of your computer’s operating system or a third-party plug-in color picker.",
                    "The Color Picker is also available when features let you choose a color. For example, by clicking the color swatch in the options bar for some tools, or the eyedroppers in some color adjustment dialog boxes.",
                    "You can choose a color by entering color component values in HSB, RGB, and Lab text boxes, or by using the color slider and the color field.",
                    "To choose a color with the color slider and color field, click in the color slider or move the color slider triangle to set one color component. Then move the circular marker or click in the color field. This sets the other two color components.",
                    "As you adjust the color using the color field and color slider, the numeric values for the different color models adjust accordingly. The rectangle to the right of the color slider displays the new color in the top half and the original color in the bottom. Alerts appear if the color is not a web-safe color&nbsp; or is out of gamut&nbsp;.",
                    "You can choose a color outside the Adobe Color Picker window. Moving the pointer over the document window changes it to the Eyedropper tool. You can then select a color by clicking in the image. The selected color is displayed in the Adobe Color Picker. You can move the Eyedropper tool anywhere on your desktop by clicking in the image and then holding down the mouse button. You can select a color by releasing the mouse button.",
                    "Using the HSB color model, the hue is specified in the color field, as an angle from 0° to 360° that corresponds to a location on the color wheel. Saturation and brightness are specified as percentages. In the color field, the hue saturation increases from left to right and the brightness increases from the bottom to top.",
                    "The color you click appears in the color slider with 0 (none of that color) at the bottom and 255 (maximum amount of that color) at the top. The color field displays the range of the other two components, one on the horizontal axis and one on the vertical axis. ",
                    "When choosing a color based on the Lab color model, the L value specifies the luminance of a color. The A value specifies how red or green a color is. The B value specifies how blue or yellow a color is.",
                    "You can choose a color by specifying a hexadecimal value that defines the R, G, and B components in a color. The three pairs of numbers are expressed in values from 00 (minimum luminance) to ff (maximum luminance). For example, 000000 is black, ffffff is white, and ff0000 is red.",
                    "The heads-up-display (HUD) color picker lets you quickly choose colors while painting in the document window, where image colors provide helpful context.",
                    " After clicking in the document window, you can release the pressed keys. Temporarily press the spacebar to maintain the selected shade while you select another hue, or vice versa.",
                    "The web‑safe colors are the 216 colors used by browsers regardless of the platform. The browser changes all colors in the image to these colors when displaying the image on an 8‑bit screen. The 216 colors are a subset of the Mac OS 8‑bit color palettes. By working only with these colors, you can be sure that art you prepare for the web will not dither on a system set to display 256 colors.",
                    "Choose Web Color Sliders from the Color panel menu. By default, web color sliders snap to web‑safe colors (indicated by tick marks) when you drag them. To override web‑safe color selection, Alt-drag (Windows) or Option-drag (Mac OS) the sliders.",
                    "If you choose a non‑web color, an alert cube  appears above the color ramp on the left side of the Color panel. Click the alert cube to select the closest web color.",
                    "Some colors in the RGB, HSB, and Lab color models cannot be printed because they are out-of-gamut and have no equivalents in the CMYK model. When you choose a non-printable color in either the Adobe Color Picker or the Color panel, a warning alert triangle appears. A swatch below the triangle displays the closest CMYK equivalent. ",
                    "The Adobe Color Picker lets you choose colors from the PANTONE MATCHING SYSTEM®, the Trumatch® Swatching System™, the Focoltone® Colour System, the Toyo Color Finder™ 1050 system, the ANPA-Color™ system, the HKS® color system, and the DIC Color Guide. ",
                    "To ensure that the final printed output is the color you want, consult your printer or service bureau and choose your color based on a printed color swatch. Manufacturers recommend that you get a new swatch book each year to compensate for fading inks and other damage. ",
                    "Photoshop prints spot colors to CMYK (process color) plates in every image mode except Duotone. To print true spot color plates, create spot color channels.",
                    "Consists of 763 CMYK colors. Focoltone colors help avoid prepress trapping and registration problems by showing the overprints that make up the colors. A swatch book with specifications for process and spot colors, overprint charts, and a chip book for marking up layouts are available from Focoltone. For more information, contact Focoltone International, Ltd., in Stafford, United Kingdom.",
                    "Used for printing projects in Europe. Each color has a specified CMYK equivalent. You can select from HKS E (for continuous stationery), HKS K (for gloss art paper), HKS N (for natural paper), and HKS Z (for newsprint). Color samplers for each scale are available. HKS Process books and swatches have been added to the color system menu.",
                    "Colors used for spot-color reproduction. The PANTONE MATCHING SYSTEM can render 1,114 colors. PANTONE color guides and chip books are printed on coated, uncoated, and matte paper stocks to ensure accurate visualization of the printed result and better on-press control. You can print a solid PANTONE color in CMYK. To compare a solid PANTONE color to its closest process color match, use the PANTONE solid to process guide. The CMYK screen tint percentages are printed under each color. For more information, contact Pantone, Inc., Carlstadt, NJ (www.pantone.com).",
                    "Consists of more than 1,000 colors based on the most common printing inks used in Japan. The TOYO Process Color Finder book and swatches have been added to the color system menu. The TOYO Color Finder 1050 Book contains printed samples of Toyo colors and is available from printers and graphic arts supply stores. For more information, contact Toyo Ink Manufacturing Co., Ltd., in Tokyo, Japan.",
                    "Provides predictable CMYK color matching with more than 2,000 achievable, computer-generated colors. Trumatch colors cover the visible spectrum of the CMYK gamut in even steps. The Trumatch Color displays up to 40 tints and shades of each hue, each originally created in four-color process and each reproducible in four colors on electronic imagesetters. In addition, four-color grays using different hues are included. For more information, contact Trumatch Inc., in New York City, New York."
                ],
                "how to choose color in toobox Adobe Photoshop")
documents5 = ([
                    "Layers\nare useful because they let you add components to an image and work\non them one at a time, without permanently changing your original\nimage. For each layer, you can adjust color and brightness, apply\nspecial effects, reposition layer content, specify opacity and blending\nvalues, and so on. You can also rearrange the stacking order, link\nlayers to work on them simultaneously, and create web animations\nwith layers.",
                    "Layers\nare like stacked, transparent sheets of glass on which you can paint images.\nYou can see through the transparent areas of a layer to the layers\nbelow. You can work on each layer independently, experimenting to\ncreate the effect you want. Each layer remains independent until\nyou combine (merge) the layers. The bottommost layer in the Layers\npanel, the Background layer, is always locked (protected), meaning\nyou cannot change its stacking order, blending mode, or opacity\n(unless you convert it into a regular layer). ",
                    "Layers are organized in the Layers panel. Keep this panel visible whenever you’re working in Adobe&nbsp;Photoshop&nbsp;Elements. With one glance, you can see the active layer (the selected layer that you are editing). You can link layers, so they move as a unit, helping you manage layers. Because multiple layers in an image increase the file size, you can reduce the file size by merging layers that you’re done editing. The Layers panel is an important source of information as you edit photos. You can also use the Layer menu to work with layers.",
                    "Enable you to fine-tune color, brightness, and saturation without\nmaking permanent changes to your image (until you flatten, or collapse, the\nadjustment layer). ",
                    "You can’t\npaint on an adjustment layer, although you can paint on its mask.\nTo paint on fill or type layers, you must first convert them into\nregular image layers.",
                    "The Layers panel (Window &gt; Layers) lists all layers in an image, from the top layer to the Background layer at the bottom. In Expert mode, if you are working in the Custom Workspace, you can drag the Layers panel out and tab it with other panels.",
                    "The active layer, or the layer that you are working on, is highlighted for easy identification. As you work in an image, check which layer is active to make sure that the adjustments and edits you perform affect the correct layer. For example, if you choose a command and nothing seems to happen, check to make sure that you’re looking at the active layer.",
                    "Using the icons in the panel, you can accomplish many tasks—such as creating, hiding, linking, locking, and deleting layers. With some exceptions, your changes affect only the selected, or active, layer, which is highlighted.",
                    "The image contains layer groups and was imported from Adobe Photoshop. Photoshop Elements doesn’t support layer groups and displays them in their collapsed state. You must simplify them to create an editable image.",
                    "Also at the top are the panel Blending Mode menu (Normal, Dissolve, Darken, and so on), an Opacity text box, and a More button displaying a menu of layer commands and panel options.",
                    "Newly\nadded layers appear above the selected layer in the Layers panel.\nYou can add layers to an image by using any of the following methods:",
                    "You can create up to 8000\nlayers in an image, each with its own blending mode and opacity.\nHowever, memory constraints may lower this limit.",
                    "To create a layer with default name and\nsettings, click the New Layer button in the Layers panel. The resulting\nlayer uses Normal mode with 100% opacity, and is named according\nto its creation order. (To rename the new layer, double-click it\nand type a new name.)",
                    "To create a layer and specify a name and options,\nchoose Layer &gt; New &gt; Layer, or choose New\nLayer from the Layers panel menu. Specify a name and other options,\nand then click OK.",
                    "The Background layer is the bottom\nlayer in an image. Other layers stack on top of the Background layer,\nwhich usually (but not always) contains the actual image data of\na photo. To protect the image, the Background layer is always locked.\nIf you want to change its stacking order, blending mode, or opacity,\nyou must first convert it into a regular layer.",
                    "Select the Background layer, and choose Duplicate Layer from the Layers panel flyout menu, to leave the Background layer intact and create a copy of it as a new layer.",
                    "You can create a duplicate layer of the converted Background layer no matter how you convert the layer; simply select the converted Background layer and choose Duplicate Layer from the Layer menu.",
                    "If you drag the Background Eraser tool\nonto the Background layer, it is automatically converted into a\nregular layer, and erased areas become transparent.",
                    "You can’t convert a layer into the Background\nlayer if the image already has a Background layer. In this case,\nyou must first convert the existing Background layer into a regular\nlayer.",
                    "Color-coding layers and groups helps you to identify related layers in the Layers panel. Simply right-click the layer or group and select a color."
                ],
                "How to create layer Adobe Photoshop")

def test(document, model=LSIModel):
    print(document[1])
    for num_topic in range(1, 20, 1):
        print("Num topic: {}".format(num_topic))
        lda = model(document[0], num_topics=num_topic)
        lda_sim = lda.similarity(document[1])
        # print(sim[0])
        for i, sim in lda_sim:
            print((i, sim))
        shutil.rmtree("models")

test(documents5, LSIModel)
exit()
lda = LSIModel(documents4[0], num_topics=len(documents4[0]))
lda_sim = lda.similarity(documents4[1])
print(documents4[0][lda_sim[0][0]])
for i, sim in lda_sim:
    print(i, sim)
