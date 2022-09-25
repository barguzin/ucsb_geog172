# Lab 1 - Google Colab, Markdown and Python

This lab will teach you how to use some important tools that you will be extensively using throughout this course: 

1. Google Colab 
2. Markdown 
3. Python 
4. Command line

---

## 1. Google Colab 

To simplify set-up, we will be using an interactive coding environment based on *Jupyter Notebooks* known as Google Colab (GC). 

### Overview 

We will start working through a series of examples that familiarize us with GC interface and work out way to markdown and Python examples. 

1. Follow [this link](https://colab.research.google.com/notebooks/basic_features_overview.ipynb) to open your first notebook and follow along with your TA, asking questions as they arise. 
2. Click 'Copy to Drive' and save the notebook to your Google Drive. To rename the notebook go to *File > Rename*. To see where the notebook was saved *File > Located in Drive*. To move the notebook go to *File > Move*.  

### Importing Libraries 

Some Python libraries are installed by default in GC (for example 'numpy'). First, follow [this link](https://colab.research.google.com/notebooks/snippets/importing_libraries.ipynb) and open the notebook. 

To check whether some package/library is installed simply run: 

````python
import numpy as np
import pandas as pd
````

If the package is installed you will see a green tick mark after executing a code cell. Now try running the following command:

````python
import geopandas as gpd
````

Chances are, you are likely to see the following text: 

````
ModuleNotFoundError                       Traceback (most recent call last)

<ipython-input-3-a62d01c1d62e> in <module>
----> 1 import geopandas as gpd

ModuleNotFoundError: No module named 'geopandas'

NOTE: If your import is failing due to a missing package, you can
manually install dependencies using either !pip or !apt.

To view examples of installing some common dependencies, click the
"Open Examples" button below.

````

This means that the package you are importing is not installed on the virtual machine that runs your Google Colab notebook. To install the package simply follow [Installation Instructions from the official GeoPandas page](https://geopandas.org/en/stable/getting_started/install.html) and run the following in your GC cell: 

````bash
!pip install geopandas shapely fiona pyproj
````

Let's digest this line above. The '!' is used to denote that we want to run the content of the cell through the command line utility (not Python). The 'pip install' command tells the computer to use package manager called 'pip' and install the following packages: 'geopandas', 'shapely', 'fiona', and 'pyproj'. Now, there is nothing special about this packages per se, but they are known as 'dependencies' and are required for 'GeoPandas' to work properly. Another Python package manager is called 'Anaconda' and takes care of the dependencies 'under the hood'. But since GC is run in the Linux environment, the default package installer for Python is 'pip' and it does not resolve dependencies during installation. 

Once the code is done running you can check whether the geopandas can be imported. 

````python
import geopandas as gpd
print(gpd.__version__)
````

### Working with files on Google Colabs 

GC can use both local files and files found online. In this activity you will practice working with files, stored on: 

* Local computer
* Mounting Google Drive locally 

1. Start by opening the corresponding [GC Notebook](https://colab.research.google.com/notebooks/io.ipynb)
2. Copy the notebook to your Drive and follow the TA instructions. 
3. You do not need to complete the entire notebook, only the first two sections outlined above. 

--- 

## 2. Markdown 

Markdown is simple, yet powerful. In fact, this entire instruction page and a course website is written almost exclusively in Markdown (+Python). 

Markdown is a lightweight markup language used for formatting elements in text documents. Markdown was created by John Gruber in 2004. Unlike **WYSIWYG** (What you see is what you get) editors like Microsoft Word, the changes in text are not visible immediately. Instead, different symbols in the text are used to denote formatting elements for example two asterisks (\*\*) are used to denote **bold font**, and pound key (\#) is used to denote Headings. See more examples in the image below: 

![](https://res.cloudinary.com/practicaldev/image/fetch/s--CRJTTGM8--/c_imagga_scale,f_auto,fl_progressive,h_900,q_auto,w_1600/https://dev-to-uploads.s3.amazonaws.com/i/g595slgphyi9lkqz2u18.png)

Obviously, markdown can be used in numerous applications: *notes, websites, books, technical documents, and academic reports*. Another reason to love markdown is the ease of adding [emojii](https://github.com/ikatyang/emoji-cheat-sheet/blob/master/README.md) to the document! 

🐊🐍🦖🐳🦗🏔️🗾🌎🏜️

You can learn more about markdown [here](https://www.markdownguide.org/getting-started/). Learning markdown is both easy and rewarding. Try creating your first markdown edited text in [Dillinger](https://dillinger.io/) right now. 

We will be using markdown to format our Google Colab notebooks. In fact, GC supports its own flavor of markdown. Now together with you TA open the [GC notebook](https://colab.research.google.com/notebooks/markdown_guide.ipynb) and follow along. 

---

## Python Basics and Plotting 

> The  canonical, "Python is a great first language", elicited, "Python is a great last language!" (Noah Spurrier)

Python is interpreted, object-oriented, high-level programming language with dynamic semantics. Python is highly suitable for Rapid Application Development, as well as glue language to connect existing components together. This makes Python easy to learn, increases its readability and reduces the cost of programming maintenance. Since Python supports packages and modules, code re-use and modularity are encouraged. In fact, it was because of rich and actively developed package ecosystem that Python has become an integral part of the scientific and computing society. 

Today Python is actively used in data analysis and data visualization. The packages, such as *matplotlib, NumPy, SciPy, Pandas and Scikit-Learn* have become standard in the data scientist toolbox. In geographical analysis, several packages (e.g. *GeoPandas, Shapely, Fiona, Folium, PySAL*) are used frequently in automated geoanalytical tasks. 

### Python Basics 

We will be following a wonderfully prepared example from the online version of the book [**"Introduction to Python for Geographic Data Analysis"**](https://pythongis.org/index.html). Please open the notebook by navigating [here](notebooks/00-python-basics.ipynb) OR go to *Notebooks > 00-python-basics*. 

Next, work with you TAs through tutorials on [FOR loops in Python](https://pythongis.org/part1/chapter-02/nb/01-for-loops.html) and [Conditional Statements](https://pythongis.org/part1/chapter-02/nb/02-conditional-statements.html). 

---

## Lab 1 - Questions 

> Please submit answers to these questions to Gauchospace prior to your lab section next week. 

1. Create Google Colab notebook 
2. Part 1 (50 points): Markdown. Use markdown to write a formatted paragraph of text about yourself. Consider including the following info
   1. Your major and intended graduation year 
   2. What made you choose the major you are currently pursuing and how this course fits into your major? 
   3. What do you intend to get out of these course? 
   4. Select one spatial problem and pose three questions that can be answered using geographical analysis? 
3. Part 2 (50 points): write a Python program that loops through the letters of alphabet and prints out every even letter. 

Please submit your lab assignment 