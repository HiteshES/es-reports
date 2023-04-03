# Import Block
import streamlit as st
import pandas as pd
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, ColumnsAutoSizeMode
import warnings
warnings.filterwarnings('ignore')

# Visualization 
import matplotlib as mpl 
import matplotlib.pyplot as plt 

# Image
from PIL import Image
import requests
from io import BytesIO

# Image
from PIL import Image
import requests
from io import BytesIO
from streamlit_echarts import st_echarts

# Data Load
df = pd.read_csv('./data/content_base.csv')

# Streamlit Formatting 
st.set_page_config(page_title= "Content Strategy", 
                   page_icon= "https://image-cdn.essentiallysports.com/wp-content/uploads/es_short_logo-1.png", layout= 'wide')

st.markdown("<h1 style='text-align: center; color: #f75453;'>Content Strategy Reports</h1>", unsafe_allow_html=True)

# st.sidebar.header("Filters")
# editors = st.sidebar.selectbox("Editors:", options= df['editor'].unique())

# Aggrid 
def aggrid_interactive_table(df: pd.DataFrame):
    """Creates an st-aggrid interactive table based on a dataframe.

    Args:
        df (pd.DataFrame]): Source dataframe

    Returns:
        dict: The selected row
    """
    options = GridOptionsBuilder.from_dataframe(
        df, enableRowGroup=True, enableValue=True, enablePivot=True
    )

    options.configure_side_bar()

    options.configure_selection("single")
    selection = AgGrid(
        df,
        enable_enterprise_modules=True,
        gridOptions=options.build(),
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
    )

    return selection

# Monthly cut
feb = df[df['month'] == 2]
mar = df[df['month'] == 3]

#Golden Metric Calculation Function
def golden_metric(df):

    # Cleaning
    df = df[df['total pageviews'] > 10000 ]
    df = df[df['engage rate % (after para1)'] > 0]
    df = df.round(2)

    domains = []
    for i in range(len(df)):
        domain = df.iloc[i]['domains']
        
        if "t.co" in domain:
            updated_domain = domain.replace("t.co", "twitter.com")
            updated_domain = ' '.join(set(updated_domain.split(',')))
            domains.append(updated_domain)
        else:
            domains.append(domain)
    
    df['domains'] = domains
    
    Pmean = df['total pageviews'].mean()
    Pstd = df['total pageviews'].std()

    # Making scroll_rate, timeonpage, engagement till para1% per pageviews
    df['scroll_rate'] = df['scroll rate']/ df['total pageviews']
    df['page_time'] = df['page time']/ df['total pageviews']
    df['Engage'] = df['engage rate % (after para1)']/ df['total pageviews']

    #Converting 2 decimal places
    df['scroll_rate'] = df['scroll rate'].round(2)
    df['page_time'] = df['page time'].round(2)
    df['Engage'] = df['engage rate % (after para1)'].round(2)
    
    
    #Normalization
    df['Pnorm'] = df['total pageviews'].apply(lambda x: (x - Pmean) / Pstd)
    df['Snorm'] = df['scroll_rate'].apply(lambda x: (x - df['scroll_rate'].min()) / (df['scroll_rate'].max() - df['scroll_rate'].min()))
    df['Tnorm'] = df['page_time'].apply(lambda x: (x - df['page_time'].min()) / (df['page_time'].max() - df['page_time'].min()))
    df['Enorm'] = df['Engage'].apply(lambda x: (x - df['Engage'].min()) / (df['Engage'].max() - df['Engage'].min()))
    
    
    # Weights
    df['derived'] = df['Snorm']*0.2 + 0.45*df['Tnorm'] + df['Enorm']*0.35
    df['golden'] = df['derived']*df['Pnorm']
    
    df.sort_values(by =['golden'], ascending=False, inplace = True)
    df.reset_index(inplace = True, drop = True)
    df.reset_index(level=0, inplace=True)
    
    # Formatting 
    columns =  df.columns
    col = []
    for name in columns:
        col.append(name.upper())
    
    df.columns = col
    df.rename(columns = {'INDEX': 'RANK'}, inplace = True)

    return df
    
# Apply golden metrics on months 

feb = golden_metric(feb)
mar = golden_metric(mar)

st.markdown("<h2 style='text-align: left;'>Current Month Analysis</h1>", unsafe_allow_html=True)

############################################################ VIZ #############################################

####Top 20 Articles
cols = ['RANK', 'TITLE', 'SPORTS', 'ENTITY', 'EDITOR', 'WRITER', 'DOMAINS','TOTAL PAGEVIEWS', 
        'SCROLL RATE', 'PAGE TIME','ENGAGE RATE % (AFTER PARA1)']

with st.expander("Top 20 Articles", expanded=False ):
    aggrid_interactive_table(mar.loc[:,cols].head(20))

# Sportwise
sports = mar['SPORTS'].unique().tolist()
sportwise = []
for sport in sports:
    sportwise.append(mar[mar['SPORTS'] == sport].sort_values(by = 'GOLDEN', ascending =False).reset_index(drop = True).iloc[:,1:12].head(5))

sportswise = pd.concat(sportwise)
sportswise.reset_index(level=0, inplace=True)
sportswise.rename(columns = {'index': 'RANK'}, inplace = True)

with st.expander("Sportwise - Top 5 Articles", expanded=False ):
    aggrid_interactive_table(sportswise.loc[:,cols])

# Editorwise
editors = mar['EDITOR'].unique().tolist()
editor = []
for e in editors:
    editor.append(mar[mar['EDITOR'] == e].sort_values(by = 'GOLDEN',  ascending =False).reset_index(drop = True).iloc[:,1:12].head(5))

editorwise = pd.concat(editor)
editorwise.reset_index(level=0, inplace=True)
editorwise.rename(columns = {'index': 'RANK'}, inplace = True)

with st.expander("Editorwise - Top 5 Articles", expanded=False ):
    aggrid_interactive_table(editorwise.loc[:,cols])

# Entitywise
entity = mar['ENTITY'].unique()
entities = []
for e in entity:
    entities.append(mar[mar['ENTITY'] == e].sort_values(by = 'GOLDEN',  ascending =False).reset_index(drop = True).iloc[:,1:12].head(5))

entitywise = pd.concat(entities)
entitywise.reset_index(level=0, inplace=True)
entitywise.rename(columns = {'index': 'RANK'}, inplace = True)

with st.expander("Entitywise - Top 5 Articles", expanded=False ):
    aggrid_interactive_table(entitywise.loc[:,cols])


sources = mar.groupby(['DOMAINS'], as_index = False)['TOTAL PAGEVIEWS'].sum().sort_values(by = 'TOTAL PAGEVIEWS', ascending = False).reset_index(drop = True).head(5)


### Source Chart
# Visualization variables initialization

background = "#404040"
text_color = "w"

mpl.rcParams['xtick.color'] = text_color
mpl.rcParams['ytick.color'] = text_color
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10

filler = "grey"
primary ="red"

title_font = "Kanit"

# creating figure and axes
fig, ax = plt.subplots(figsize=(16,7))
fig.set_facecolor(background)
ax.patch.set_facecolor(background)

# adding a grid with zorder and style
ax.grid(ls="dotted",lw="0.3",color="lightgrey", zorder= 1, visible = True)

# Create data for pie chart
sizes = sources['TOTAL PAGEVIEWS']
labels = sources['DOMAINS']
colors = ['#F94144', '#F8961E', '#F9C74F', '#90BE6D', '#43AA8B']

# set the width and height of the chart
width = 0.2
height = 0.3

# set the position of the chart within the axis
left = 0.1
bottom = 0.1

# Create pie chart with logo
text_props = {'color': 'white', 'fontsize': 12}
ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, radius=0.8, counterclock=False,
       textprops=text_props)

# Add legend
ax.legend(labels, title="Sources", loc="lower right", bbox_to_anchor=(1, 0, 0.5, 1))

# For badge: Adding new axes
ax2 = fig.add_axes([0.04,0.99,0.1,0.14])
ax2.axis("off")

url = "https://image-cdn.essentiallysports.com/wp-content/uploads/es_short_logo-1.png"
response = requests.get(url)
img = Image.open(BytesIO(response.content))
ax2.imshow(img)     

# Adding Credits
fig.text(0.05, -0.028, "Data powered by content strategy", fontstyle="italic", fontsize=12, 
         color=text_color)

plt.tight_layout()

st.markdown("<h4> Top 5 sources</h4>", unsafe_allow_html=True)

st.pyplot(plt.gcf())