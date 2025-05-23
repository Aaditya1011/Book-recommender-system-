import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

CLIENT_ID = 'SPOTIFY-CLIENT-ID'
CLIENT_SECRET = 'SPOTIFY-SECRET-KEY'

auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID,client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)

songs = pd.read_csv('songs.csv')
song_name = songs['SongName'].to_list()

image_urls = []
for i in range(len(song_name)):
    print(f'fetching data for',song_name[i])
    result = sp.search(q=song_name[i],type='track',limit=1)
    temp_url = result['tracks']['items'][0]['album']['images'][0]['url']
    image_urls.append((song_name[i],temp_url))

urls = pd.DataFrame(image_urls)
url_csv = urls.to_csv('url_csv')

