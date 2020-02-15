from keras.datasets import mnist
from sklearn.preprocessing import StandardScaler
from matplotlib.pyplot import imshow
from bokeh.io import output_notebook, show, reset_output
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool
from matplotlib import cm
from PIL import Image
from skimage import io, filters, measure, morphology, img_as_ubyte
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import base64

def b64_image_files(images, colormap='magma'):
    cmap = cm.get_cmap(colormap)
    urls = []
    for im in images:
        png = to_png(img_as_ubyte(cmap(im)))
        url = 'data:image/png;base64,' + base64.b64encode(png).decode('utf-8')
        urls.append(url)
    return urls

def to_png(arr):
    out = BytesIO()
    im = Image.fromarray(arr)
    im.save(out, format='png')
    return out.getvalue()

def getTestTrainSamples():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train_nsamples, x_train_nx, x_train_ny = x_train.shape
    x_test_nsamples, x_test_nx, x_test_ny = x_test.shape

    train_numbers = x_train
    test_numbers = x_test

    x_train = x_train.reshape((x_train_nsamples, x_train_nx * x_train_ny))
    x_test = x_test.reshape((x_test_nsamples, x_test_nx * x_test_ny))

    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255

    return x_train, y_train, x_test, y_test, train_numbers, test_numbers

def visualizeSamples(x_train, y_train, x_test, y_test):
    f, a = plt.subplots(2, 10, figsize=(20, 5))

    for i in range(10):
        a[0][i].imshow(np.reshape(x_train[i], (28, 28)), interpolation="nearest")
        a[0][i].set_title("Number: " + str(y_train[i]))
        a[0][i].axis('off')
        f.show()
        
    for i in range(10):
        a[1][i].imshow(np.reshape(x_test[i], (28, 28)), interpolation="nearest")
        a[1][i].set_title("Number: " + str(y_test[i]))
        a[1][i].axis('off')
        f.show()

def visualize2DMapPCA(num, pComponentsDf_train, numeros_train):
    colormap = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728', 4: '#9467bd', 5: '#8c564b',
            6: '#e377c2', 7: '#7f7f7f', 8: '#bcbd22', 9: '#17becf'}

    colores = [colormap[x] for x in pComponentsDf_train.loc[ :num, "y"]]

    files = b64_image_files(numeros_train[:num])

    source = ColumnDataSource(data=dict(
        x=pComponentsDf_train.loc[:num, "pcomp_1"],
        y=pComponentsDf_train.loc[:num, "pcomp_2"],
        imgs=files,
        color=colores    
    ))

    TOOLTIPS = """
        <div>
            <div>
                <img
                    src="@imgs" height="40" width="40"
                    style="float: left; margin: 0px 15px 15px 0px;"
                    border="2"
                ></img>
            </div>
        </div>
    """

    p = figure(plot_width=950, plot_height=950, title="Mouse over the dots")
    hover = HoverTool()
    hover.tooltips = TOOLTIPS
    p.tools.append(hover)        
    p.circle(x='x', y='y', source=source, color='color', fill_alpha=0.2, size=10)
    show(p)


def visualize2DMapMLP(num, pComponentsDf_train, numeros_train, encoded):
    colormap = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728', 4: '#9467bd', 5: '#8c564b',
            6: '#e377c2', 7: '#7f7f7f', 8: '#bcbd22', 9: '#17becf'}

    colores = [colormap[x] for x in pComponentsDf_train.loc[ :num, "y"]]

    files = b64_image_files(numeros_train[:num])

    source = ColumnDataSource(data=dict(
        x=encoded.loc[:num, 0],
        y=encoded.loc[:num, 1],
        imgs=files,
        color=colores    
    ))

    TOOLTIPS = """
        <div>
            <div>
                <img
                    src="@imgs" height="40" width="40"
                    style="float: left; margin: 0px 15px 15px 0px;"
                    border="2"
                ></img>
            </div>
        </div>
    """

    p = figure(plot_width=950, plot_height=950, title="Mouse over the dots")
    hover = HoverTool()
    hover.tooltips = TOOLTIPS
    p.tools.append(hover)            
    p.circle(x='x', y='y', source=source, color='color', fill_alpha=0.2, size=10)
    show(p)   

def visualize2DMapConvNet(num, pComponentsDf_train, numeros_train):
    colormap = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728', 4: '#9467bd', 5: '#8c564b',
           6: '#e377c2', 7: '#7f7f7f', 8: '#bcbd22', 9: '#17becf'}

    colores = [colormap[x] for x in pComponentsDf_train.loc[ :num, "y"]]

    files = b64_image_files(numeros_train[:num])

    source = ColumnDataSource(data=dict(
        x=aa.loc[:num, 0],
        y=aa.loc[:num, 1],
        imgs=files,
        color=colores    
    ))

    TOOLTIPS = """
        <div>
            <div>
                <img
                    src="@imgs" height="40" width="40"
                    style="float: left; margin: 0px 15px 15px 0px;"
                    border="2"
                ></img>
            </div>
        </div>
    """

    p = figure(plot_width=950, plot_height=950, title="Mouse over the dots")
    hover = HoverTool()
    hover.tooltips = TOOLTIPS
    p.tools.append(hover)
    p.circle(x='x', y='y', source=source, color='color', fill_alpha=0.2, size=10)
    show(p)


def Plot2D(Num=1000, Images, X, Y, Numbers):
    colormap = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728', 4: '#9467bd', 5: '#8c564b',
        6: '#e377c2', 7: '#7f7f7f', 8: '#bcbd22', 9: '#17becf'}

    colors = [colormap[x] for x in Numbers[:Num]]
    value = [x for x in Numbers[:Num]]

    files = b64_image_files(train_numbers[:Num])

    source = ColumnDataSource(Images=dict(
        x=X,
        y=Y,
        imgs=files,
        color=colors,
        value=value
    ))

    TOOLTIPS = """
        <div>
            <div>
                <img src="@imgs" height="40" width="40"
                    style="float: left; margin: 0px 15px 15px 0px;"
                    border="2"/>                
                <b>@value</b>            
            </div>
        </div>
    """

    p = figure(plot_width=950, plot_height=950)
    hover = HoverTool()
    hover.tooltips = TOOLTIPS
    p.tools.append(hover)        
    p.circle(x='x', y='y', source=source, color='color', fill_alpha=0.2, size=10)
    show(p)    