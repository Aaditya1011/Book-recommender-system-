from flask import Flask,render_template,request
import pickle
import numpy as np


popular_new = pickle.load(open('popular_new.pkl','rb'))
pt = pickle.load(open('pt.pkl','rb'))
books = pickle.load(open('books.pkl','rb'))
similarity_scores = pickle.load(open('similarity_scores.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',
    book_name = list(popular_new['Book-Title'].values),
    author = list(popular_new['Book-Author'].values),
    image = list(popular_new['Image-URL-M'].values),
    votes = list(popular_new['num_rating'].values),
    rating = list(popular_new['avg_rating'].values))

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/credits')
def credits():
    return render_template('credits.html')

@app.route('/recommend_books',methods=['post'])
def recommend():
    user_input = request.form.get('user_input')
    
    # find index of the book name.
    #index = np.where(pt.index == user_input)[0][0]
    index = np.where(pt.index.str.lower() == user_input.lower())[0][0]

    # get similar items.
    similar_items = sorted(list(enumerate(similarity_scores[index])),key=lambda x:x[1],reverse=True)[1:6]
    
    # add the names to data.
    data = []
    for i in similar_items:
        items = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        items.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        items.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        items.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        
        data.append(items)

    return render_template('recommend.html',data=data)

if __name__ == '__main__':
    app.run(debug=True)