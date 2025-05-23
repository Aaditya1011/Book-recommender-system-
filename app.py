from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd

# load book data.
popular_new = pickle.load(open('models/book_model/popular_new.pkl','rb'))
pt = pickle.load(open('models/book_model/pt.pkl','rb'))
books = pickle.load(open('models/book_model/books.pkl','rb'))
similarity_scores = pickle.load(open('models/book_model/similarity_scores.pkl','rb'))

# load song data.
songs = pd.read_csv('data/songs.csv')
song_kmeans = pickle.load(open('models/song_model/kmean.pkl','rb'))
song_scaler = pickle.load(open('models/song_model/song_scaler.pkl','rb'))
song_vectorizer = pickle.load(open('models/song_model/song_vectorizer.pkl','rb'))

# initialise Flask app.
app = Flask(__name__)

@app.route('/')
def home():
    popular_songs = songs.sort_values(by='Popularity',ascending=False)

    return render_template('home.html',
    book_name = list(popular_new['Book-Title'].values),
    author = list(popular_new['Book-Author'].values),
    image = list(popular_new['Image-URL-M'].values),
    votes = list(popular_new['num_rating'].values),
    rating = list(popular_new['avg_rating'].values),
    song_name = list(popular_songs['SongName'].values),
    song_img = list(popular_songs['URL'].values),
    song_artist = list(popular_songs['ArtistName'].values),
    song_popularity = list(popular_songs['Popularity'].values))

@app.route('/recommend',methods=['get','post'])
def recommend_ui():
    return render_template('recommend.html')

@app.route('/credits')
def credits():
    return render_template('credits.html')

@app.route('/recommend_books',methods=['post'])
def recommend():
    user_input = request.form.get('user_input')
    
    # find index of the book name.
    index = np.where(pt.index.str.lower() == user_input.lower())[0]
        # index = np.where(pt.index == user_input)[0][0]

    # if no index is found, give a random index.
    if len(index) == 1:
        index = index[0]
    else:
        index = np.random.randint(0,679)

    # get similar items.
    similar_items = sorted(list(enumerate(similarity_scores[index])),key=lambda x:x[1],reverse=True)[1:13]
    
    # add the names to data.
    data = []
    for i in similar_items:
        items = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        items.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        items.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        items.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        
        data.append(items)

    return render_template('recommend.html',data=data,text='Recommended Books.')

@app.route('/recommend_songs',methods=['post'])
def recommend_song():
    # get user input and find the index of the song.
    user_input = request.form.get('user_input')
    index = np.where(songs['SongName'].apply(lambda x:x.lower()) == user_input.lower())[0]

    # if no index is found, give a random index.
    if len(index) == 1:
        index = index[0]
    else:
        index = np.random.randint(0,1000)

    # get songs using index and select required features.
    new_songs = songs.drop(['Key','Mode'],axis=1) 
    sample = np.array(new_songs.iloc[index])

    # get artist name and convert it into vector.
    song_artist = sample[1]
    artist_name = song_vectorizer.transform([song_artist]).toarray()

    # get audio features and scale it.
    audio_features = sample[2:-3].reshape(1,-1)
    scaled_features = song_scaler.transform(audio_features)

    # concatinate both arrays to get final input.
    final_input = np.hstack((artist_name,scaled_features))

    # get the prediction.
    prediction = song_kmeans.predict(final_input)

    # get similar songs from same cluster.
    similar_songs = songs[songs['cluster'] == prediction[0]]
    # similar_songs = similar_songs.sort_values('Popularity',ascending=False)

    # select required attributes (name,artist,image-link) of the songs.
    similar_songs.reset_index(drop=True,inplace=True)
    data = list(similar_songs[['SongName','ArtistName','URL']].values[:12])

    return render_template('recommend.html',data=data,text='Recommended Songs.')



if __name__ == '__main__':
    app.run(debug=True)