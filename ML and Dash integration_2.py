####### Importing the libraries
import pickle
import pandas as pd
import webbrowser
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from matplotlib import pyplot as plt


from collections import Counter
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import wordcloud
import os
####### Declaring Global variables
project_name =  None
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
scrappedReviews = None
scrappedReviews
ideas=['CHOOSE YOUR OPTION',"I said they are somewhat small",'Nothing like the real thing with real leather.','All you can ask for. I got what i ordered Thanks!!' ,
       'quality was good','quality was bad'," ",'Great material and perfect quality!']

####### Defining My Functions
def jaya():
        pass;
        
def loading_model():
    
    scrappedReviews = pd.read_csv('C://Users//user//Downloads//scrappedRe.csv')
 
    global pickle_model
    file = open("C://Users/user//Downloads//pickle_model (3).pkl", 'rb') 
    pickle_model = pickle.load(file)
    
    

    global vocab
    file = open("C://Users//user//Downloads//features.pkl(3)", 'rb') 
    vocab = pickle.load(file)

    temp = []
    for i in scrappedReviews['reviews']:
        temp.append(check_review(i)[0])
    scrappedReviews['sentiment'] = temp
    
    positive = len(scrappedReviews[scrappedReviews['sentiment']==1])
    negative = len(scrappedReviews[scrappedReviews['sentiment']==0])
    
    explode = (0.1,0)  

    langs = ['Positive', 'Negative',]
    students = [positive,negative]
    colors = ['#41fc1c','red']
    plt.pie(students,explode=explode,startangle=90,colors=colors, labels = langs,autopct='%1.2f%%')
    cwd = os.getcwd()
    if 'assets' not in os.listdir(cwd):
        os.makedirs(cwd+'/assets')
    plt.savefig('assets/sentiment.png')
    #wordcloud
    dataset = scrappedReviews['reviews'].to_list()
    str1 = ''
    for i in dataset:
        str1 = str1+i
    str1 = str1.lower()

    stopwords = set(STOPWORDS)
    cloud = WordCloud(width = 800, height = 400,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(str1)
    cloud.to_file("assets/wordCloud.png")
    #drop down
    global chart_dropdown_values
    chart_dropdown_values = {}
    for i in range(400,501):
        chart_dropdown_values[scrappedReviews['reviews'][i]] = scrappedReviews['reviews'][i]
    chart_dropdown_values = [{"label":key, "value":values} for key,values in chart_dropdown_values.items()]
    
def check_review(reviewText):

    #load the vectorize and call transform and then pass that to model preidctor

    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vocab)
    vectorised_review = transformer.fit_transform(loaded_vec.fit_transform([reviewText]))


    return pickle_model.predict(vectorised_review)

def create_app_ui():
    global project_name
    main_layout = html.Div(
    [
    html.H1(id='Main_title', children = "Sentiment Analysis with Insights",style={'text-align':'center','font-family'}),
    html.Hr(style={'background-color':'black'}),
    
    html.H2(children = "Pie Chart",style = {'text-align':'center','text-decoration':'underline'}),
    
    html.P([html.Img(src=app.get_asset_url('sentiment.png'),style={'width':'700px','height':'400px'})],style={'text-align':'center'}),
    
    html.Hr(style={'background-color':'black'}),
    
    html.H2(children = "WordCloud",style = {'text-align':'center','text-decoration':'underline'}),
    
    html.P([html.Img(src=app.get_asset_url('wordCloud.png'),style={'width':'700px','height':'400px'})],style={'text-align':'center'}),
    
    html.Hr(style={'background-color':'black'}),
    
    html.H2(children = "Select a Review",style = {'text-align':'center','text-decoration':'underline'}),
    
    dcc.Dropdown(
                id='Chart_Dropdown', 
                  options=chart_dropdown_values,
                  placeholder = 'Select a Review',style={'font-size':'22px','height':'70px'}
                    ),
    html.H1(children = 'Missing',id='sentiment1',style={'text-align':'center'}),
    html.Hr(style={'background-color':'black'}),
    html.H2(children = "Find Sentiment of Your Review",style = {'text-align':'center','text-decoration':'underline'}),
    dcc.Textarea(
        id = 'textarea_review',
        placeholder = 'Enter the review here.....',
        style = {'width':'100%', 'height':150,'font-size':'22px'}
        ),
    
    dbc.Button(
        children = 'FInd Review',
        id = 'button_review',
        color = 'dark',
        style= {'width':'100%'}
        ),
    
    html.H1(children = 'Missing', id='result',style={'text-align':'center'})
    
    ]    
    )
    
    return main_layout


def browser_opening():
    webbrowser.open_new('http://127.0.0.1:8050/')
    
@app.callback(
    Output('result', 'children'),
    [
    Input('button', 'n_clicks')
    ],
    [
    State('textarea', 'value')
    ]
    )    
def update_app_ui(n_clicks, textarea):
    result_list = check_review(textarea)
    
    if (result_list[0] == 0 ):
        return dbc.Alert("Negative", color="danger")
    elif (result_list[0] == 1 ):
        return dbc.Alert("Positive", color="success")
    else:
        return dbc.Alert("Unknown", color="dark")
    
@app.callback(
    Output('result1', 'children'),
    [
    Input('button', 'n_clicks')
    ],
    [
     State('dropdown', 'value')
     ]
    )
def update_dropdown(n_clicks, value):
    result_list = check_review(value)
    
    if (result_list[0] == 0 ):
        return dbc.Alert("Negative", color="danger")
    elif (result_list[0] == 1 ):
        return dbc.Alert("Positive", color="success")
    else:
        return dbc.Alert("Unknown", color="dark")
        

####### Main Function to control the Flow of your Project
def main():
    print("Start of project")
    global project_name
    global app
    jaya()
    loading_model()
    browser_opening()
    
    
    
    project_name = "Sentiment Analysis with Insights"
    
    app.title = project_name
    app.layout = create_app_ui()
    app.run_server()
    
    print("End of project")
    project_name = None
    
    app = None
        
####### Calling the main function 
if __name__ == '__main__':
    main()