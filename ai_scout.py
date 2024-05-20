# -*- coding: utf-8 -*-
"""ai_scout.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1wAuQDZcRUKZfqM78XA5blYN3-BqVQlkr
"""

!pip install openAI

import requests
import pandas as pd

from datetime import datetime

from openai import OpenAI
from bs4 import BeautifulSoup

client = OpenAI(
    api_key="..."
)

# player_name = 'Francisco Conceição'
# url = 'https://fbref.com/en/players/5ef3d210/Francisco-Conceicao'
# attrs_percentile_stats = 'scout_summary_AM'
# attrs_sim_players = 'similar_AM'

player_name = 'Francisco Trincão'
url = 'https://fbref.com/en/players/77e39b04/Francisco-Trincao'
attrs_percentile_stats = 'scout_summary_AM' #MF' #DF
attrs_sim_players = 'similar_AM'

df = pd.read_html(
    url,
    attrs={'id': attrs_percentile_stats}
)[0]

df1 = pd.read_html(
    url,
    attrs={'id': attrs_sim_players}
)[0]

df1['Player_Club'] = df1['Player'] + ' (' + df1['Squad'] + ')'

sim_players = list(df1['Player_Club'][:6])
sim_players

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

position = soup.select_one('p:-soup-contains("Position")').text.split(':')[-2].split(' ')[0].strip()
birthday = soup.select_one('span[id="necro-birth"]').text.strip()

height = soup.select('#meta > div > p:nth-child(3) > span:nth-child(1)')[0].text.split('<span>')[0]
weight = soup.select('#meta > div > p:nth-child(3) > span:nth-child(2)')[0].text
foot = soup.select_one('p:-soup-contains("Footed")').text.split('Footed')[1].split(': ')[1]

age = (datetime.now() - datetime.strptime(birthday, '%B %d, %Y')).days // 365
team = soup.select_one('p:-soup-contains("Club")').text.split(':')[-1].strip()

media_item_div = soup.find('div', {'class': 'media-item'})
img_tag = media_item_div.find('img') if media_item_div else None

player_image_url = img_tag['src'] if img_tag else 'URL default da imagem se não encontrada'

# df.columns = df.columns.droplevel(0)

df = df.dropna(subset='Statistic')

df.head()

prompt = f"""
I need you to create a scouting report on {player_name}. Can you provide me with a summary of their strengths and weaknesses?

Here is the data I have on him:

Here is the data I have on him:

Player: {player_name}
Height: {height}
Weight: {weight}
Position: {position}
Age: {age}
Team: {team}

List of similar players to {player_name} and respective clubs.
{df1.to_markdown()}

{df.to_markdown()}

Return the scouting report in the following HTML:


<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Report for {player_name}</title>
</head>
<body>
  <div style="position: relative; padding: 30px; display: flex; align-items: center;">
    <div style="flex: 0 0 auto;">
      <img src="{player_image_url}" alt=" {player_name} headshot">
    </div>
    <div style="margin-left: 20px;">
      <h1 style="color: darkblue; font-size: 28px;">Report for {player_name}</h1>
      <p>
        <span style="font-weight: bold;">Player:</span> {player_name}<br>
        <span style="font-weight: bold;">Height:</span> {height}<br>
        <span style="font-weight: bold;">Weight:</span> {weight}<br>
        <span style="font-weight: bold;">Position:</span> {position}<br>
        <span style="font-weight: bold;">Age:</span> {age}<br>
        <span style="font-weight: bold;">Team:</span> {team}
      </p>
    </div>
    <div style="flex: 1;"></div>
  </div>
  <div style="padding: 0 30px;">
    <h1 style="color: darkblue; text-align: center;">Summary</h1>
    <p>
      <a brief summary of the player's overall performance and if he would be beneficial to the team>
    </p>
    <h2 style="color: darkblue; text-align: left;">Strengths</h2>
    <ul>
      <li><i>a list of 1 to 3 strengths</i></li>
    </ul>
    <h2 style="color: darkblue; text-align: left;">Weaknesses</h2>
    <ul>
      <li><i>a list of 1 to 3 weaknesses</i></li>
    </ul>
    <h2 style="color: darkblue; text-align: left;">Potential</h2>
    <p>
      < assessment of the player's potential for growth >
    </p>
    <h2 style="color: darkblue; text-align: center;">Similar players</h2>
      <p> < mention the similar players to {player_name} ></p>
    <div style="text-align: center; margin-top: 50px;">
      <img src="https://raw.githubusercontent.com/ricardoandreom/Data/main/Images/Personal%20Logos/Half%20Space%20Preto.png" alt="Logo" style="width: 200px;">
    </div>
  </div>
</body>
</html>

"""

print(prompt)

response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {"role": "system", "content": "You are a professional football (soccer) scout."},
        {"role": "user", "content": prompt},
    ],
)

#print(response.choices[0].message.content)

!pip install weasyprint

from weasyprint import HTML

html = response.choices[0].message.content#.split('```html')[1].split('```')[0]

output_pdf = player_name + "_scouting_report.pdf"

with open("temp.html", "w") as f:
    f.write(html)

HTML("temp.html").write_pdf(output_pdf)

import os
os.remove("temp.html")

response.choices[0].message.content#.split('```html')[1].split('```')[0]

#response.choices[0].message.content.split('```html')[1].split('```')[0]