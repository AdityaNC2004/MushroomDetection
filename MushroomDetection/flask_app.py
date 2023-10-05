import os

cwd = os.getcwd()

import colorsys
import io
import json
import pickle
import re
from base64 import b64encode

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import seaborn as sns
from flask import Flask, redirect, render_template, request, url_for
from PIL import Image 
import io
from skimage.io import imread
from skimage.transform import resize
from werkzeug.utils import secure_filename

# Create Flask app
app = Flask(__name__)

# Set up upload folder for images and set maximum content length

app.config['UPLOAD_FOLDER'] = os.path.join(cwd, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# static_folder = os.path.join(cwd, 'static')
# images_folder = os.path.join(static_folder, 'images')
# if not os.path.exists(static_folder):
#         os.makedirs(static_folder)
# if not os.path.exists(images_folder):
#         os.makedirs(images_folder)

app.static_folder = os.path.join(cwd, 'static')

# Home page route
@app.route('/')
def temp():
        return render_template('first_page.html')

# Route for handling form submission
@app.route('/',methods=['POST','GET'])
def get_input():
        if request.method == 'POST':
                # Retrieve image file from form data
                file = request.files['imageInput']
                filename = secure_filename(file.filename)

                # Save image to upload folder and redirect to prediction route
                file_path = re.sub(r'[\\]','/', os.path.join(app.config['UPLOAD_FOLDER']))
                file.save(file_path)
                
                return redirect(url_for('run_pred',image=filename))

# Route for prediction
@app.route('/run_pred/<image>')
def run_pred(image):

        pillow_img = Image.open(os.path.join(cwd, 'uploads'))
        image_io = io.BytesIO()
        pillow_img.save(image_io, 'PNG')
        image_link_input = 'data:image/png;base64,' + b64encode(image_io.getvalue()).decode('ascii')

        # Load first order model for mushroom species prediction
        first_order_model_path = os.path.join(cwd, 'mushroom_models', 'mushroom_model', 'mushroom_model')
        first_order_model_path = re.sub(r'[\\]', '/', first_order_model_path)
        first_order_model = keras.models.load_model(first_order_model_path)

        # Load image and preprocess for prediction
        image_data = imread(os.path.join(cwd, 'uploads'), as_gray=False)
        img = image_data / 255
        img = np.array(img)
        img = np.array(resize(img, (112,112,3)))
        img = np.expand_dims(img,0)

        # Get first order prediction
        predictions = first_order_model.predict(img, verbose=0)
        first_order_pred = np.argmax(predictions)

        # Load pickled dictionaries and lists
        pickle_path = re.sub(r'[\\]', '/', os.path.join(cwd, 'pickle_jar', 'order_dict.pickle'))
        with open(pickle_path, 'rb') as f:
                order_dict = pickle.load(f)

        pickle_path_inverse_dict = re.sub(r'[\\]', '/', os.path.join(cwd, 'pickle_jar', 'species_dict_inverse.pickle'))
        with open(pickle_path_inverse_dict, 'rb') as f:
                species_dict_inverse = pickle.load(f)

        pickle_path_species_conversion_dict = re.sub(r'[\\]', '/', os.path.join(cwd, 'pickle_jar', 'species_conversion_dict.pickle'))
        with open(pickle_path_species_conversion_dict, 'rb') as f:
                species_conversion_dict = pickle.load(f)

        pickle_path_order_dict = re.sub(r'[\\]', '/', os.path.join(cwd, 'pickle_jar', 'order_dict.pickle'))
        with open(pickle_path_order_dict, 'rb') as f:
                order_dict = pickle.load(f)

        pickle_path_metadata_dict = re.sub(r'[\\]', '/', os.path.join(cwd, 'pickle_jar', 'species_metadata_dict.pickle'))
        with open(pickle_path_metadata_dict, 'rb') as f:
                species_metadata_dict = pickle.load(f)

        pickle_path_images_links = re.sub(r'[\\]', '/', os.path.join(cwd, 'pickle_jar', 'species_images_links.pickle'))
        with open(pickle_path_images_links, 'rb') as f:
                species_images_dict = pickle.load(f)

        pickle_path_edibility_dict = re.sub(r'[\\]', '/', os.path.join(cwd, 'pickle_jar', 'edibility_dict.pickle'))
        with open(pickle_path_edibility_dict, 'rb') as f:
                edibility_dict = pickle.load(f)

        pickle_path_spotting_list = re.sub(r'[\\]', '/', os.path.join(cwd, 'pickle_jar', 'spotting_list.pickle'))
        with open(pickle_path_spotting_list, 'rb') as f:
                spotting_list = pickle.load(f)

        colour_pack = [
        '#636EFA',  # muted blue
        '#00CC96',  # cooked asparagus green
        '#d62728',  # brick red
        '#9467bd',  # muted purple
        '#e377c2',  # raspberry yogurt pink
        '#17becf'   # blue-teal
        ]
        colour_dict = {'#636EFA': (99,110,250),
                        '#00CC96': (0,204,150),
                        '#d62728': (214,39,40),
                        '#9467bd': (148,103,189),
                        '#e377c2': (227,119,194),
                        '#17becf': (23,190,207)}
        colour_list = []
        colour_num = np.random.randint(0,len(colour_pack))
        colour_pick = colour_pack[colour_num]
        colour_list.append(colour_pick)

        # If first order prediction corresponds to a specific order, use second order model for species prediction
        if first_order_pred in [0,2,3,5,6,8,10]:
                second_order_model_path = os.path.join(cwd, 'mushroom_models', f'mushroom_model_order_{first_order_pred}', f'mushroom_model_order_{first_order_pred}')
                second_order_model_path = re.sub(r'[\\]', '/', second_order_model_path)
                second_order_model = keras.models.load_model(second_order_model_path)
                predictions = second_order_model.predict(img, verbose=0)
                second_order_pred = np.argmax(predictions)
                species_name = species_dict_inverse[species_conversion_dict[first_order_pred][second_order_pred]]
        # If first order prediction corresponds to a species, return species name
        else:
                species_name = order_dict[first_order_pred]

        # Get mushroom image link
        image_link = species_images_dict[species_name]

        # Define mushroom edibility
        edibility = edibility_dict[species_name]

        # Generate pie chart
        metadata_df = species_metadata_dict[species_name]
        continent_group = metadata_df[metadata_df['continent']!='NA_'].groupby('continent').count()['id']
        continent_data = list(continent_group.values)
        continent_labels = list(continent_group.index)
        continent_df = pd.DataFrame(continent_data, index=continent_labels, columns=['count'])
        continent_fig = px.pie(continent_df, 
                                values='count', 
                                names=continent_df.index, 
                                title=f'Continental Distribution of {species_name} samples')
        continent_fig_json = json.dumps(continent_fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Generate sample type chart
        sample_group = metadata_df[metadata_df['sample_type']!='NA_'].groupby('sample_type').count()['id']
        sample_data = list(sample_group.values)
        sample_labels = list(sample_group.index)
        sample_df = pd.DataFrame(sample_data, index=sample_labels, columns=['count'])
        sample_fig = px.pie(sample_df, 
                                values='count', 
                                names=sample_df.index, 
                                title=f'Sample Type Distribution of {species_name}') 
        sample_fig_json = json.dumps(sample_fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Generate biome chart
        biome_group = metadata_df[metadata_df['Biome']!='NA_'].groupby('Biome').count()['id']
        biome_data = list(biome_group.values)
        biome_labels = list(biome_group.index)
        biome_df = pd.DataFrame(biome_data, index=biome_labels, columns=['count'])
        biome_fig = px.pie(biome_df, 
                                values='count', 
                                names=biome_df.index, 
                                title=f'Biome Distribution of {species_name} samples') 
        biome_fig_json = json.dumps(biome_fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Generate MAT chart
        mat_df = metadata_df[metadata_df['MAT']!='NA_']['MAT']
        mat_df = mat_df.astype('float')
        mat_df = mat_df.sort_values(ascending=True)
        mat_low = int(np.floor(np.min(np.array(mat_df.values))))
        mat_high = int(np.ceil(np.max(np.array(mat_df.values))))
        mat_range = (mat_high-mat_low)
        if mat_range > 50 and mat_range <= 100:
                mat_range = mat_range//5
        elif mat_range > 100 and mat_range <= 500:
                mat_range = mat_range//10
        elif mat_range > 500:
                mat_range = mat_range//100
        else:
                mat_range = mat_range

        colour = colour_dict[colour_list[0]]
        r, g, b = colour
        h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
        palette = []
        palette_n = len(mat_df)//3+1
        for i in range(-palette_n, palette_n):
                new_l = max(0, min(1, l + i/(len(mat_df)*2)))
                new_l = [new_l if new_l<10 else 10][0]
                r, g, b = tuple(round(x*255) for x in colorsys.hls_to_rgb(h, new_l, s))
                palette.append(f'rgb{tuple([r,g,b])}')
        palette.reverse()
        
        if mat_range <= 1:
                palette = colour_list
                
        mat_fig = px.histogram(mat_df, x='MAT', 
                                nbins=mat_range, 
                                title=f'MAT (Mean Annual Temperature) of {species_name} samples',
                                labels={"MAT": "MAT (°C)",
                                        "count": "Count"},
                                color='MAT',
                                color_discrete_sequence=palette) 
        mat_fig.update_layout(showlegend=False)
        mat_fig_json = json.dumps(mat_fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Generate MAP chart
        map_df = metadata_df[metadata_df['MAP']!='NA_']['MAP']
        map_df = map_df.astype('float')
        map_df = map_df.sort_values(ascending=True)
        map_low = int(np.floor(np.min(np.array(map_df.values))))
        map_high = int(np.ceil(np.max(np.array(map_df.values))))
        map_range = (map_high-mat_low)
        if map_range > 10 and map_range <= 100:
                map_range = map_range//5
        elif map_range > 100 and map_range <= 500:
                map_range = map_range//10
        elif map_range > 500:
                map_range = map_range//100
        else:
                map_range = map_range

        colour = colour_dict[colour_list[0]]
        r, g, b = colour
        h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
        palette = []
        palette_n = len(map_df)//3+1
        for i in range(-palette_n, palette_n):
                new_l = max(0, min(1, l + i/(len(map_df)*2)))
                new_l = [new_l if new_l<10 else 10][0]
                r, g, b = tuple(round(x*255) for x in colorsys.hls_to_rgb(h, new_l, s))
                palette.append(f'rgb{tuple([r,g,b])}')
        palette.reverse()
        
        if map_range <= 1:
                palette = colour_list
                
        map_fig = px.histogram(map_df, x='MAP', 
                                nbins=map_range, 
                                title=f'MAP (Mean Annual Precipitation) of {species_name} samples',
                                labels={"MAP": "MAP (mm)",
                                        "count": "Count"},
                                color='MAP',
                                color_discrete_sequence=palette)
        map_fig.update_layout(showlegend=False)
        map_fig_json = json.dumps(map_fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Generate PH chart
        try:
                pH_df = metadata_df[metadata_df['pH']!='NA_']['pH']
                pH_df = pH_df.astype('float')
                pH_df = pH_df.sort_values(ascending=True)
                pH_low = int(np.floor(np.min(np.array(pH_df.values))))
                pH_high = int(np.ceil(np.max(np.array(pH_df.values))))
                pH_range = pH_high-pH_low

                colour = colour_dict[colour_list[0]]
                r, g, b = colour
                h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
                palette = []
                palette_n = len(pH_df)//3+1
                for i in range(-palette_n, palette_n):
                        new_l = max(0, min(1, l + i/(len(pH_df)*2)))
                        new_l = [new_l if new_l<10 else 10][0]
                        r, g, b = tuple(round(x*255) for x in colorsys.hls_to_rgb(h, new_l, s))
                        palette.append(f'rgb{tuple([r,g,b])}')
                        palette.reverse()
                
                if pH_range <= 1:
                        palette = colour_list

                pH_fig = px.histogram(pH_df, x='pH', 
                                        nbins=pH_range+1, 
                                        title=f'pH Distribution of {species_name} samples',
                                        labels={"pH": "pH",
                                                "count": "Count"},
                                        color='pH',
                                        color_discrete_sequence=palette
                                        )
                pH_fig.update_layout(showlegend=False)
                pH_fig_json = json.dumps(pH_fig, cls=plotly.utils.PlotlyJSONEncoder)
        except: # Two species don't have pH data
                pH_fig_json = None

        # Generate abundances chart
        abundances_df = metadata_df[metadata_df['abundances']!='NA_']['abundances']
        abundances_df = abundances_df.sort_values(ascending=True)
        abundances_low = int(np.floor(np.min(np.array(abundances_df.values))))
        abundances_high = int(np.ceil(np.max(np.array(abundances_df.values))))
        abundances_range = (abundances_high-abundances_low)
        if abundances_range > 10 and abundances_range <= 100:
                abundances_range = abundances_range//5
        elif abundances_range > 100 and abundances_range <= 500:
                abundances_range = abundances_range//10
        elif abundances_range > 500 and abundances_range <= 1000:
                abundances_range = abundances_range//100
        elif abundances_range > 1000 and abundances_range <= 10000:
                abundances_range = abundances_range//100
        elif abundances_range > 10000:
                abundances_range = abundances_range//1000
        else:
                abundances_range = abundances_range

        colour = colour_dict[colour_list[0]]
        r, g, b = colour
        h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
        palette = []
        palette_n = len(abundances_df)//3+1
        for i in range(-palette_n, palette_n):
                new_l = max(0, min(1, l + i/(len(abundances_df)*2)))
                new_l = [new_l if new_l<10 else 10][0]
                r, g, b = tuple(round(x*255) for x in colorsys.hls_to_rgb(h, new_l, s))
                palette.append(f'rgb{tuple([r,g,b])}')
        palette.reverse()
        
        if abundances_range <= 1:
                palette = colour_list

        abundances_fig = px.histogram(abundances_df, x='abundances', 
                                nbins=abundances_range, 
                                title=f'Abundances (Number of Mushrooms) of {species_name} samples',
                                labels={"abundances": "Abundances",
                                        "count": "Count"},
                                color='abundances',
                                color_discrete_sequence=palette)
        abundances_fig.update_layout(showlegend=True)
        abundances_fig_json = json.dumps(abundances_fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Generate geo scatterplot
        geo_data = metadata_df[(metadata_df['longitude']!='NA_') & (metadata_df['latitude']!='NA_')]
        geo_df = geo_data[['longitude','latitude']]
        geo_fig = px.scatter_geo(geo_df, 
                                        lat='latitude', 
                                        lon='longitude', 
                                        title=f'Geographical Location of {species_name} samples',
                                color_discrete_sequence=colour_list)
        geo_fig_json = json.dumps(geo_fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Get spottings data
        species_spotting = []
        spotting_images = {}
        spotting_geo_figs = {}
        spotting_date = {}
        spotting_location = {}
        for s, spotting in enumerate(spotting_list):
                if spotting['species_name'] == species_name:
                        species_spotting.append(spotting)

        if len(species_spotting) > 0:
                for s, spotting in enumerate(species_spotting):
                        if s < 10:
                                spotting_images[f'spotting_images_{s}'] = spotting['image_link']
                                spotting_date[f'spotting_date_{s}'] = spotting['date']
                                spotting_location[f'spotting_location_{s}'] = spotting['location']
                        
                                spotting_data = {'latitude': [spotting['lat']], 'longitude': [spotting['lon']]}
                                spotting_df = pd.DataFrame(spotting_data)
                                spotting_geo_fig = px.scatter_geo(spotting_df, 
                                                                lat='latitude', 
                                                                lon='longitude', 
                                                                title=f'Spotting Location of {species_name}',
                                                        color_discrete_sequence=colour_list)
                                spotting_geo_fig_json = json.dumps(spotting_geo_fig, cls=plotly.utils.PlotlyJSONEncoder)
                                spotting_geo_figs[f'spotting_geo_fig_{s}'] = spotting_geo_fig_json

                spotting_data_all = {'latitude': [spotting['lat'] for spotting in species_spotting], 'longitude': [spotting['lon'] for spotting in species_spotting]}
                spotting_all_df = pd.DataFrame(spotting_data_all)
                spotting_geo_fig_all = px.scatter_geo(spotting_all_df, 
                                                lat='latitude', 
                                                lon='longitude', 
                                                title=f'All Spotting Locations of {species_name}',
                                                color_discrete_sequence=colour_list)
                spotting_geo_fig_all_json = json.dumps(spotting_geo_fig_all, cls=plotly.utils.PlotlyJSONEncoder)

                spotting_country = {'country': list(set([spotting['location'].split(',')[-1].upper() for spotting in species_spotting]))}
                spotting_country_df = pd.DataFrame(spotting_country)

                colour = colour_dict[colour_list[0]]
                r, g, b = colour
                h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
                palette = []
                palette_n = len(spotting_country_df)//2+1
                try:
                        for i in range(-palette_n, palette_n):
                                new_l = max(0, min(1, l + i/len(pH_df)))
                                new_l = [new_l if new_l<10 else 10][0]
                                r, g, b = tuple(round(x*255) for x in colorsys.hls_to_rgb(h, new_l, s))
                                palette.append(f'rgb{tuple([r,g,b])}')
                except:
                        palette = colour_list

                spotting_country_fig = fig = px.choropleth(spotting_country_df,
                                                                locations='country', 
                                                                locationmode='country names',
                                                                color='country', 
                                                                color_continuous_scale=palette)
                spotting_country_fig.update_layout(title=f'Countries of {species_name} Spottings')
                spotting_country_fig_json = json.dumps(spotting_country_fig, cls=plotly.utils.PlotlyJSONEncoder)

                spotting_country_pie = {'country': [spotting['location'].split(',')[-1].upper() for spotting in species_spotting],
                                        'count': ['count' for spotting in species_spotting]}
                spotting_country_pie_df = pd.DataFrame(spotting_country_pie)
                spotting_country_group_df = spotting_country_pie_df.groupby('country').count()
                spotting_country_pie_fig = px.pie(spotting_country_group_df, 
                                                        values='count', 
                                                        names=spotting_country_group_df.index, 
                                                        title=f'Spotting Country Distribution of {species_name}')
                spotting_country_pie_fig_json = json.dumps(spotting_country_pie_fig, cls=plotly.utils.PlotlyJSONEncoder)

                if len(spotting_images) == 10:
                        return render_template('results.html',
                                                image_link_input=image_link_input,
                                                image_link=image_link,
                                                image_data=b64encode(image_data).decode('utf-8'),
                                                edibility=edibility,
                                                continent_fig=continent_fig_json, 
                                                sample_fig=sample_fig_json, 
                                                biome_fig=biome_fig_json, 
                                                mat_fig=mat_fig_json,
                                                map_fig=map_fig_json,
                                                pH_fig=pH_fig_json,
                                                abundances_fig=abundances_fig_json, 
                                                geo_fig=geo_fig_json, 
                                                species_name=species_name, 
                                                colour=colour_pick,
                                                spotting_geo_fig_all=spotting_geo_fig_all_json,
                                                spotting_country_fig=spotting_country_fig_json,
                                                spotting_country_pie_fig=spotting_country_pie_fig_json,
                                                spotting_images_0=spotting_images['spotting_images_0'],
                                                spotting_images_1=spotting_images['spotting_images_1'],
                                                spotting_images_2=spotting_images['spotting_images_2'],
                                                spotting_images_3=spotting_images['spotting_images_3'],
                                                spotting_images_4=spotting_images['spotting_images_4'],
                                                spotting_images_5=spotting_images['spotting_images_5'],
                                                spotting_images_6=spotting_images['spotting_images_6'],
                                                spotting_images_7=spotting_images['spotting_images_7'],
                                                spotting_images_8=spotting_images['spotting_images_8'],
                                                spotting_images_9=spotting_images['spotting_images_9'],
                                                spotting_date_0=spotting_date['spotting_date_0'],
                                                spotting_date_1=spotting_date['spotting_date_1'],
                                                spotting_date_2=spotting_date['spotting_date_2'],
                                                spotting_date_3=spotting_date['spotting_date_3'],
                                                spotting_date_4=spotting_date['spotting_date_4'],
                                                spotting_date_5=spotting_date['spotting_date_5'],
                                                spotting_date_6=spotting_date['spotting_date_6'],
                                                spotting_date_7=spotting_date['spotting_date_7'],
                                                spotting_date_8=spotting_date['spotting_date_8'],
                                                spotting_date_9=spotting_date['spotting_date_9'],
                                                spotting_location_0=spotting_location['spotting_location_0'],
                                                spotting_location_1=spotting_location['spotting_location_1'],
                                                spotting_location_2=spotting_location['spotting_location_2'],
                                                spotting_location_3=spotting_location['spotting_location_3'],
                                                spotting_location_4=spotting_location['spotting_location_4'],
                                                spotting_location_5=spotting_location['spotting_location_5'],
                                                spotting_location_6=spotting_location['spotting_location_6'],
                                                spotting_location_7=spotting_location['spotting_location_7'],
                                                spotting_location_8=spotting_location['spotting_location_8'],
                                                spotting_location_9=spotting_location['spotting_location_9'],
                                                spotting_geo_fig_0=spotting_geo_figs['spotting_geo_fig_0'],
                                                spotting_geo_fig_1=spotting_geo_figs['spotting_geo_fig_1'],
                                                spotting_geo_fig_2=spotting_geo_figs['spotting_geo_fig_2'],
                                                spotting_geo_fig_3=spotting_geo_figs['spotting_geo_fig_3'],
                                                spotting_geo_fig_4=spotting_geo_figs['spotting_geo_fig_4'],
                                                spotting_geo_fig_5=spotting_geo_figs['spotting_geo_fig_5'],
                                                spotting_geo_fig_6=spotting_geo_figs['spotting_geo_fig_6'],
                                                spotting_geo_fig_7=spotting_geo_figs['spotting_geo_fig_7'],
                                                spotting_geo_fig_8=spotting_geo_figs['spotting_geo_fig_8'],
                                                spotting_geo_fig_9=spotting_geo_figs['spotting_geo_fig_9'],
                                                )
                elif len(spotting_images) == 9:
                        return render_template('results.html',
                                                image_link_input=image_link_input,
                                                image_link=image_link,
                                                image_data=b64encode(image_data).decode('utf-8'),
                                                edibility=edibility,
                                                continent_fig=continent_fig_json, 
                                                sample_fig=sample_fig_json, 
                                                biome_fig=biome_fig_json, 
                                                mat_fig=mat_fig_json,
                                                map_fig=map_fig_json,
                                                pH_fig=pH_fig_json,
                                                abundances_fig=abundances_fig_json, 
                                                geo_fig=geo_fig_json, 
                                                species_name=species_name, 
                                                colour=colour_pick,
                                                spotting_geo_fig_all=spotting_geo_fig_all_json,
                                                spotting_country_fig=spotting_country_fig_json,
                                                spotting_country_pie_fig=spotting_country_pie_fig_json,
                                                spotting_images_0=spotting_images['spotting_images_0'],
                                                spotting_images_1=spotting_images['spotting_images_1'],
                                                spotting_images_2=spotting_images['spotting_images_2'],
                                                spotting_images_3=spotting_images['spotting_images_3'],
                                                spotting_images_4=spotting_images['spotting_images_4'],
                                                spotting_images_5=spotting_images['spotting_images_5'],
                                                spotting_images_6=spotting_images['spotting_images_6'],
                                                spotting_images_7=spotting_images['spotting_images_7'],
                                                spotting_images_8=spotting_images['spotting_images_8'],
                                                spotting_date_0=spotting_date['spotting_date_0'],
                                                spotting_date_1=spotting_date['spotting_date_1'],
                                                spotting_date_2=spotting_date['spotting_date_2'],
                                                spotting_date_3=spotting_date['spotting_date_3'],
                                                spotting_date_4=spotting_date['spotting_date_4'],
                                                spotting_date_5=spotting_date['spotting_date_5'],
                                                spotting_date_6=spotting_date['spotting_date_6'],
                                                spotting_date_7=spotting_date['spotting_date_7'],
                                                spotting_date_8=spotting_date['spotting_date_8'],
                                                spotting_location_0=spotting_location['spotting_location_0'],
                                                spotting_location_1=spotting_location['spotting_location_1'],
                                                spotting_location_2=spotting_location['spotting_location_2'],
                                                spotting_location_3=spotting_location['spotting_location_3'],
                                                spotting_location_4=spotting_location['spotting_location_4'],
                                                spotting_location_5=spotting_location['spotting_location_5'],
                                                spotting_location_6=spotting_location['spotting_location_6'],
                                                spotting_location_7=spotting_location['spotting_location_7'],
                                                spotting_location_8=spotting_location['spotting_location_8'],
                                                spotting_geo_fig_0=spotting_geo_figs['spotting_geo_fig_0'],
                                                spotting_geo_fig_1=spotting_geo_figs['spotting_geo_fig_1'],
                                                spotting_geo_fig_2=spotting_geo_figs['spotting_geo_fig_2'],
                                                spotting_geo_fig_3=spotting_geo_figs['spotting_geo_fig_3'],
                                                spotting_geo_fig_4=spotting_geo_figs['spotting_geo_fig_4'],
                                                spotting_geo_fig_5=spotting_geo_figs['spotting_geo_fig_5'],
                                                spotting_geo_fig_6=spotting_geo_figs['spotting_geo_fig_6'],
                                                spotting_geo_fig_7=spotting_geo_figs['spotting_geo_fig_7'],
                                                spotting_geo_fig_8=spotting_geo_figs['spotting_geo_fig_8'],
                                                )
                elif len(spotting_images) == 8:
                        return render_template('results.html',
                                                image_link_input=image_link_input,
                                                image_link=image_link,
                                                image_data=b64encode(image_data).decode('utf-8'),
                                                edibility=edibility,
                                                continent_fig=continent_fig_json, 
                                                sample_fig=sample_fig_json, 
                                                biome_fig=biome_fig_json, 
                                                mat_fig=mat_fig_json,
                                                map_fig=map_fig_json,
                                                pH_fig=pH_fig_json,
                                                abundances_fig=abundances_fig_json, 
                                                geo_fig=geo_fig_json, 
                                                species_name=species_name, 
                                                colour=colour_pick,
                                                spotting_geo_fig_all=spotting_geo_fig_all_json,
                                                spotting_country_fig=spotting_country_fig_json,
                                                spotting_country_pie_fig=spotting_country_pie_fig_json,
                                                spotting_images_0=spotting_images['spotting_images_0'],
                                                spotting_images_1=spotting_images['spotting_images_1'],
                                                spotting_images_2=spotting_images['spotting_images_2'],
                                                spotting_images_3=spotting_images['spotting_images_3'],
                                                spotting_images_4=spotting_images['spotting_images_4'],
                                                spotting_images_5=spotting_images['spotting_images_5'],
                                                spotting_images_6=spotting_images['spotting_images_6'],
                                                spotting_images_7=spotting_images['spotting_images_7'],
                                                spotting_date_0=spotting_date['spotting_date_0'],
                                                spotting_date_1=spotting_date['spotting_date_1'],
                                                spotting_date_2=spotting_date['spotting_date_2'],
                                                spotting_date_3=spotting_date['spotting_date_3'],
                                                spotting_date_4=spotting_date['spotting_date_4'],
                                                spotting_date_5=spotting_date['spotting_date_5'],
                                                spotting_date_6=spotting_date['spotting_date_6'],
                                                spotting_date_7=spotting_date['spotting_date_7'],
                                                spotting_location_0=spotting_location['spotting_location_0'],
                                                spotting_location_1=spotting_location['spotting_location_1'],
                                                spotting_location_2=spotting_location['spotting_location_2'],
                                                spotting_location_3=spotting_location['spotting_location_3'],
                                                spotting_location_4=spotting_location['spotting_location_4'],
                                                spotting_location_5=spotting_location['spotting_location_5'],
                                                spotting_location_6=spotting_location['spotting_location_6'],
                                                spotting_location_7=spotting_location['spotting_location_7'],
                                                spotting_geo_fig_0=spotting_geo_figs['spotting_geo_fig_0'],
                                                spotting_geo_fig_1=spotting_geo_figs['spotting_geo_fig_1'],
                                                spotting_geo_fig_2=spotting_geo_figs['spotting_geo_fig_2'],
                                                spotting_geo_fig_3=spotting_geo_figs['spotting_geo_fig_3'],
                                                spotting_geo_fig_4=spotting_geo_figs['spotting_geo_fig_4'],
                                                spotting_geo_fig_5=spotting_geo_figs['spotting_geo_fig_5'],
                                                spotting_geo_fig_6=spotting_geo_figs['spotting_geo_fig_6'],
                                                spotting_geo_fig_7=spotting_geo_figs['spotting_geo_fig_7'],
                                                )
                elif len(spotting_images) == 7:
                        return render_template('results.html',
                                                image_link_input=image_link_input,
                                                image_link=image_link,
                                                image_data=b64encode(image_data).decode('utf-8'),
                                                edibility=edibility,
                                                continent_fig=continent_fig_json, 
                                                sample_fig=sample_fig_json, 
                                                biome_fig=biome_fig_json, 
                                                mat_fig=mat_fig_json,
                                                map_fig=map_fig_json,
                                                pH_fig=pH_fig_json,
                                                abundances_fig=abundances_fig_json, 
                                                geo_fig=geo_fig_json, 
                                                species_name=species_name, 
                                                colour=colour_pick,
                                                spotting_geo_fig_all=spotting_geo_fig_all_json,
                                                spotting_country_fig=spotting_country_fig_json,
                                                spotting_country_pie_fig=spotting_country_pie_fig_json,
                                                spotting_images_0=spotting_images['spotting_images_0'],
                                                spotting_images_1=spotting_images['spotting_images_1'],
                                                spotting_images_2=spotting_images['spotting_images_2'],
                                                spotting_images_3=spotting_images['spotting_images_3'],
                                                spotting_images_4=spotting_images['spotting_images_4'],
                                                spotting_images_5=spotting_images['spotting_images_5'],
                                                spotting_images_6=spotting_images['spotting_images_6'],
                                                spotting_date_0=spotting_date['spotting_date_0'],
                                                spotting_date_1=spotting_date['spotting_date_1'],
                                                spotting_date_2=spotting_date['spotting_date_2'],
                                                spotting_date_3=spotting_date['spotting_date_3'],
                                                spotting_date_4=spotting_date['spotting_date_4'],
                                                spotting_date_5=spotting_date['spotting_date_5'],
                                                spotting_date_6=spotting_date['spotting_date_6'],
                                                spotting_location_0=spotting_location['spotting_location_0'],
                                                spotting_location_1=spotting_location['spotting_location_1'],
                                                spotting_location_2=spotting_location['spotting_location_2'],
                                                spotting_location_3=spotting_location['spotting_location_3'],
                                                spotting_location_4=spotting_location['spotting_location_4'],
                                                spotting_location_5=spotting_location['spotting_location_5'],
                                                spotting_location_6=spotting_location['spotting_location_6'],
                                                spotting_geo_fig_0=spotting_geo_figs['spotting_geo_fig_0'],
                                                spotting_geo_fig_1=spotting_geo_figs['spotting_geo_fig_1'],
                                                spotting_geo_fig_2=spotting_geo_figs['spotting_geo_fig_2'],
                                                spotting_geo_fig_3=spotting_geo_figs['spotting_geo_fig_3'],
                                                spotting_geo_fig_4=spotting_geo_figs['spotting_geo_fig_4'],
                                                spotting_geo_fig_5=spotting_geo_figs['spotting_geo_fig_5'],
                                                spotting_geo_fig_6=spotting_geo_figs['spotting_geo_fig_6'],
                                                )
                elif len(spotting_images) == 6:
                        return render_template('results.html',
                                                image_link_input=image_link_input,
                                                image_link=image_link,
                                                image_data=b64encode(image_data).decode('utf-8'),
                                                edibility=edibility,
                                                continent_fig=continent_fig_json, 
                                                sample_fig=sample_fig_json, 
                                                biome_fig=biome_fig_json, 
                                                mat_fig=mat_fig_json,
                                                map_fig=map_fig_json,
                                                pH_fig=pH_fig_json,
                                                abundances_fig=abundances_fig_json, 
                                                geo_fig=geo_fig_json, 
                                                species_name=species_name, 
                                                colour=colour_pick,
                                                spotting_geo_fig_all=spotting_geo_fig_all_json,
                                                spotting_country_fig=spotting_country_fig_json,
                                                spotting_country_pie_fig=spotting_country_pie_fig_json,
                                                spotting_images_0=spotting_images['spotting_images_0'],
                                                spotting_images_1=spotting_images['spotting_images_1'],
                                                spotting_images_2=spotting_images['spotting_images_2'],
                                                spotting_images_3=spotting_images['spotting_images_3'],
                                                spotting_images_4=spotting_images['spotting_images_4'],
                                                spotting_images_5=spotting_images['spotting_images_5'],
                                                spotting_date_0=spotting_date['spotting_date_0'],
                                                spotting_date_1=spotting_date['spotting_date_1'],
                                                spotting_date_2=spotting_date['spotting_date_2'],
                                                spotting_date_3=spotting_date['spotting_date_3'],
                                                spotting_date_4=spotting_date['spotting_date_4'],
                                                spotting_date_5=spotting_date['spotting_date_5'],
                                                spotting_location_0=spotting_location['spotting_location_0'],
                                                spotting_location_1=spotting_location['spotting_location_1'],
                                                spotting_location_2=spotting_location['spotting_location_2'],
                                                spotting_location_3=spotting_location['spotting_location_3'],
                                                spotting_location_4=spotting_location['spotting_location_4'],
                                                spotting_location_5=spotting_location['spotting_location_5'],
                                                spotting_geo_fig_0=spotting_geo_figs['spotting_geo_fig_0'],
                                                spotting_geo_fig_1=spotting_geo_figs['spotting_geo_fig_1'],
                                                spotting_geo_fig_2=spotting_geo_figs['spotting_geo_fig_2'],
                                                spotting_geo_fig_3=spotting_geo_figs['spotting_geo_fig_3'],
                                                spotting_geo_fig_4=spotting_geo_figs['spotting_geo_fig_4'],
                                                spotting_geo_fig_5=spotting_geo_figs['spotting_geo_fig_5'],
                                                )
                elif len(spotting_images) == 5:
                        return render_template('results.html',
                                                image_link_input=image_link_input,
                                                image_link=image_link,
                                                image_data=b64encode(image_data).decode('utf-8'),
                                                edibility=edibility,
                                                continent_fig=continent_fig_json, 
                                                sample_fig=sample_fig_json, 
                                                biome_fig=biome_fig_json, 
                                                mat_fig=mat_fig_json,
                                                map_fig=map_fig_json,
                                                pH_fig=pH_fig_json,
                                                abundances_fig=abundances_fig_json, 
                                                geo_fig=geo_fig_json, 
                                                species_name=species_name, 
                                                colour=colour_pick,
                                                spotting_geo_fig_all=spotting_geo_fig_all_json,
                                                spotting_country_fig=spotting_country_fig_json,
                                                spotting_country_pie_fig=spotting_country_pie_fig_json,
                                                spotting_images_0=spotting_images['spotting_images_0'],
                                                spotting_images_1=spotting_images['spotting_images_1'],
                                                spotting_images_2=spotting_images['spotting_images_2'],
                                                spotting_images_3=spotting_images['spotting_images_3'],
                                                spotting_images_4=spotting_images['spotting_images_4'],
                                                spotting_date_0=spotting_date['spotting_date_0'],
                                                spotting_date_1=spotting_date['spotting_date_1'],
                                                spotting_date_2=spotting_date['spotting_date_2'],
                                                spotting_date_3=spotting_date['spotting_date_3'],
                                                spotting_date_4=spotting_date['spotting_date_4'],
                                                spotting_location_0=spotting_location['spotting_location_0'],
                                                spotting_location_1=spotting_location['spotting_location_1'],
                                                spotting_location_2=spotting_location['spotting_location_2'],
                                                spotting_location_3=spotting_location['spotting_location_3'],
                                                spotting_location_4=spotting_location['spotting_location_4'],
                                                spotting_geo_fig_0=spotting_geo_figs['spotting_geo_fig_0'],
                                                spotting_geo_fig_1=spotting_geo_figs['spotting_geo_fig_1'],
                                                spotting_geo_fig_2=spotting_geo_figs['spotting_geo_fig_2'],
                                                spotting_geo_fig_3=spotting_geo_figs['spotting_geo_fig_3'],
                                                spotting_geo_fig_4=spotting_geo_figs['spotting_geo_fig_4'],
                                                )
                elif len(spotting_images) == 4:
                        return render_template('results.html',
                                                image_link_input=image_link_input,
                                                image_link=image_link,
                                                image_data=b64encode(image_data).decode('utf-8'),
                                                edibility=edibility,
                                                continent_fig=continent_fig_json, 
                                                sample_fig=sample_fig_json, 
                                                biome_fig=biome_fig_json, 
                                                mat_fig=mat_fig_json,
                                                map_fig=map_fig_json,
                                                pH_fig=pH_fig_json,
                                                abundances_fig=abundances_fig_json, 
                                                geo_fig=geo_fig_json, 
                                                species_name=species_name, 
                                                colour=colour_pick,
                                                spotting_geo_fig_all=spotting_geo_fig_all_json,
                                                spotting_country_fig=spotting_country_fig_json,
                                                spotting_country_pie_fig=spotting_country_pie_fig_json,
                                                spotting_images_0=spotting_images['spotting_images_0'],
                                                spotting_images_1=spotting_images['spotting_images_1'],
                                                spotting_images_2=spotting_images['spotting_images_2'],
                                                spotting_images_3=spotting_images['spotting_images_3'],
                                                spotting_date_0=spotting_date['spotting_date_0'],
                                                spotting_date_1=spotting_date['spotting_date_1'],
                                                spotting_date_2=spotting_date['spotting_date_2'],
                                                spotting_date_3=spotting_date['spotting_date_3'],
                                                spotting_location_0=spotting_location['spotting_location_0'],
                                                spotting_location_1=spotting_location['spotting_location_1'],
                                                spotting_location_2=spotting_location['spotting_location_2'],
                                                spotting_location_3=spotting_location['spotting_location_3'],
                                                spotting_geo_fig_0=spotting_geo_figs['spotting_geo_fig_0'],
                                                spotting_geo_fig_1=spotting_geo_figs['spotting_geo_fig_1'],
                                                spotting_geo_fig_2=spotting_geo_figs['spotting_geo_fig_2'],
                                                spotting_geo_fig_3=spotting_geo_figs['spotting_geo_fig_3'],
                                                )
                elif len(spotting_images) == 3:
                        return render_template('results.html',
                                                image_link_input=image_link_input,
                                                image_link=image_link,
                                                image_data=b64encode(image_data).decode('utf-8'),
                                                edibility=edibility,
                                                continent_fig=continent_fig_json, 
                                                sample_fig=sample_fig_json, 
                                                biome_fig=biome_fig_json, 
                                                mat_fig=mat_fig_json,
                                                map_fig=map_fig_json,
                                                pH_fig=pH_fig_json,
                                                abundances_fig=abundances_fig_json, 
                                                geo_fig=geo_fig_json, 
                                                species_name=species_name, 
                                                colour=colour_pick,
                                                spotting_geo_fig_all=spotting_geo_fig_all_json,
                                                spotting_country_fig=spotting_country_fig_json,
                                                spotting_country_pie_fig=spotting_country_pie_fig_json,
                                                spotting_images_0=spotting_images['spotting_images_0'],
                                                spotting_images_1=spotting_images['spotting_images_1'],
                                                spotting_images_2=spotting_images['spotting_images_2'],
                                                spotting_date_0=spotting_date['spotting_date_0'],
                                                spotting_date_1=spotting_date['spotting_date_1'],
                                                spotting_date_2=spotting_date['spotting_date_2'],
                                                spotting_location_0=spotting_location['spotting_location_0'],
                                                spotting_location_1=spotting_location['spotting_location_1'],
                                                spotting_location_2=spotting_location['spotting_location_2'],
                                                spotting_geo_fig_0=spotting_geo_figs['spotting_geo_fig_0'],
                                                spotting_geo_fig_1=spotting_geo_figs['spotting_geo_fig_1'],
                                                spotting_geo_fig_2=spotting_geo_figs['spotting_geo_fig_2'],
                                                )
                elif len(spotting_images) == 2:
                        return render_template('results.html',
                                                image_link_input=image_link_input,
                                                image_link=image_link,
                                                image_data=b64encode(image_data).decode('utf-8'),
                                                edibility=edibility,
                                                continent_fig=continent_fig_json, 
                                                sample_fig=sample_fig_json, 
                                                biome_fig=biome_fig_json, 
                                                mat_fig=mat_fig_json,
                                                map_fig=map_fig_json,
                                                pH_fig=pH_fig_json,
                                                abundances_fig=abundances_fig_json, 
                                                geo_fig=geo_fig_json, 
                                                species_name=species_name, 
                                                colour=colour_pick,
                                                spotting_geo_fig_all=spotting_geo_fig_all_json,
                                                spotting_country_fig=spotting_country_fig_json,
                                                spotting_country_pie_fig=spotting_country_pie_fig_json,
                                                spotting_images_0=spotting_images['spotting_images_0'],
                                                spotting_images_1=spotting_images['spotting_images_1'],
                                                spotting_date_0=spotting_date['spotting_date_0'],
                                                spotting_date_1=spotting_date['spotting_date_1'],
                                                spotting_location_0=spotting_location['spotting_location_0'],
                                                spotting_location_1=spotting_location['spotting_location_1'],
                                                spotting_geo_fig_0=spotting_geo_figs['spotting_geo_fig_0'],
                                                spotting_geo_fig_1=spotting_geo_figs['spotting_geo_fig_1'],
                                                )
                elif len(spotting_images) == 1:
                        return render_template('results.html',
                                                image_link_input=image_link_input,
                                                image_link=image_link,
                                                image_data=b64encode(image_data).decode('utf-8'),
                                                edibility=edibility,
                                                continent_fig=continent_fig_json, 
                                                sample_fig=sample_fig_json, 
                                                biome_fig=biome_fig_json, 
                                                mat_fig=mat_fig_json,
                                                map_fig=map_fig_json,
                                                pH_fig=pH_fig_json,
                                                abundances_fig=abundances_fig_json, 
                                                geo_fig=geo_fig_json, 
                                                species_name=species_name, 
                                                colour=colour_pick,
                                                spotting_geo_fig_all=spotting_geo_fig_all_json,
                                                spotting_country_fig=spotting_country_fig_json,
                                                spotting_country_pie_fig=spotting_country_pie_fig_json,
                                                spotting_images_0=spotting_images['spotting_images_0'],
                                                spotting_date_0=spotting_date['spotting_date_0'],
                                                spotting_location_0=spotting_location['spotting_location_0'],
                                                spotting_geo_fig_0=spotting_geo_figs['spotting_geo_fig_0'],
                                                )
        else:
                return render_template('results.html',
                                        image_link_input=image_link_input,
                                        image_link=image_link,
                                        image_data=b64encode(image_data).decode('utf-8'),
                                        edibility=edibility,
                                        continent_fig=continent_fig_json, 
                                        sample_fig=sample_fig_json, 
                                        biome_fig=biome_fig_json, 
                                        mat_fig=mat_fig_json,
                                        map_fig=map_fig_json,
                                        pH_fig=pH_fig_json,
                                        abundances_fig=abundances_fig_json, 
                                        geo_fig=geo_fig_json, 
                                        species_name=species_name, 
                                        colour=colour_pick)

# Run Flask app
if __name__ == '__main__':
        app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)