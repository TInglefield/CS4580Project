#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
CS 4580 - Assignment 5. Titanic Crew Analysis
"""
import sys
import os
import requests
import random
import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

# Dataset and folder hidden for public access

def download_dataset(url, data_file, data_folder=DATA_FOLDER):
    """
    Downloads a dataset from a specified URL and saves it to a local directory.
    Parameters:
    url (str): The base URL where the dataset is hosted.
    data_file (str): The name of the dataset file to be downloaded.
    data_folder (str): The name of the data folder to store data
    Returns:
    None
    Side Effects:
    - Creates a directory if it does not exist.
    - Downloads the dataset file and saves it to the specified directory.
    - Prints messages indicating the status of the download process.
    Notes:
    - If the dataset file already exists in the specified directory, the function will not download it again.
    - If the download fails, an error message will be printed.
    """

    # Check if the data folder exists and file_path is a valid path
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print(f'Created folder {data_folder}')
    # Check if the file already exists
    data_folder = os.path.join(data_folder, data_file)
    if os.path.exists(data_folder):
        print(f'Dataset {data_file} already exists in {data_folder}')
        return

    # Include file name to url server address
    url = f'{url}/{data_file}'
    # Download the dataset from the server to DATA_FOLDER
    response = requests.get(url)
    if response.status_code == 200:
        with open(data_folder, 'wb') as f:
            f.write(response.content)
        print(f'Downloaded dataset {data_file} to {data_folder}')
    else:
        print(f'Error downloading dataset {data_file} from {url}')


def load_data(file_path):
    """
    Load data from a CSV file into a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    DataFrame: Returns a pandas DataFrame if the file exists and is a valid CSV file.

    Raises:
    ValueError: If the file is not a valid CSV file.
    FileNotFoundError: If the file does not exist.
    """
    # Check if file is csv format
    if not file_path.endswith('.csv'):
        print(f'File {file_path} is not a valid CSV file')
        raise ValueError
    # Check if data is a valid file path or raise an error
    if not os.path.exists(file_path):
        print(f'File {file_path} does not exist')
        raise FileNotFoundError

    # Load the data into a DataFrame
    df = pd.read_csv(file_path)
    return df


class MovieRecommendationEngine:
    """ 
    Program Class Flow
    Welcome Message: 
        Start with a brief welcome message to explain the purpose of the recommendation engine.
    Initial Recommendations: 
        Display a list of 10 random movies from the dataset when the program starts.
    Search Feature: 
        Implement a search feature allowing the user to filter movies based on either the release year or keywords in the title.
    Selection Tracking: 
        Allow the user to select movies they are interested in from the recommendations. Track all selected movies and display a list of these selections.
    Dynamic Recommendations: 
        After each movie selection, provide a fresh set of recommendations based on past selections.
    Recommendations:
        Based on the user’s selection history, factoring in release year, title keywords, or genres.
    Finally: 
        The user should have the ability to select the number of recommendations (K value) the engine provides.
    """

    def __init__(self, data, k=30):
        self.data = data
        self.k = k
        self.selected_movies = []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.model = None
        self._prepare_model()
        self.prev_recommendations = []
        self.current_recommendations = []

    # Welcome Message
    def welcome_message(self):
        print("\nWelcome to the Movie Recommendation Engine!")
        print(
            "Here, you can discover movies based on your preferences and past selections.\n")

    # Initial Recommendations
    def initial_recommendations(self):
        print()
        print("Initial Recommendations:")
        initial_recommendations = self.data.sample(10)
        self.prev_recommendations = initial_recommendations
        for idx, row in initial_recommendations.iterrows():
            print(f"{row['title']} (IMDb ID: {row['imdbId']})")

    # K Nearest Neighbor (KNN) Algorithm
    def _prepare_model(self):
        self.data['combined_features'] = self.data['title'] + \
            " " + self.data['genres']
        features = self.vectorizer.fit_transform(
            self.data['combined_features'])
        self.model = NearestNeighbors(metric='cosine', algorithm='brute')
        self.model.fit(features)
        self.features = features

    def recommend(self, selected_movie_title=None):
        """
        Recommend movies based solely on KNN results for user-selected movies.
        """
        # Add selected movie to the list of selected movies if provided
        if selected_movie_title:
            matching_titles = self.data[self.data['title'].str.lower(
            ).str.startswith(selected_movie_title.lower())]
            if not matching_titles.empty:
                self.selected_movies.extend(matching_titles['title'].tolist())
            else:
                print(f"\nNo movies found starting with '{selected_movie_title}'.")

        print(f"\nRecommendations based on your Search using KNN model ({len(self.selected_movies)} movies):")

        # Get vector for selected movies if any
        if self.selected_movies:
            # Find the indices for selected movies
            selected_indices = self.data[self.data['title'].isin(
                self.selected_movies)].index
            # Convert the selected movies' combined features to vectors
            selected_vectors = self.vectorizer.transform(
                self.data.iloc[selected_indices]['combined_features'])
            # Use the KNN model to find similar movies
            _, indices = self.model.kneighbors(
                selected_vectors, n_neighbors=self.k + 10)  # Getting extra results
            # Retrieve recommended movies using the indices
            recommendations = self.data.iloc[indices.flatten(
            )].drop_duplicates()

            # Limit recommendations to top k results and display them
            recommendations = recommendations.head(self.k)
            for idx, row in recommendations.iterrows():
                print(f"{row['title']} (IMDb ID: {row['imdbId']})")
        else:
            # If no selected movies, return random recommendations
            recommendations = self.data.sample(self.k)
            for idx, row in recommendations.iterrows():
                print(f"{row['title']} (IMDb ID: {row['imdbId']})")

        self.prev_recommendations = self.current_recommendations
        self.current_recommendations = recommendations
        return recommendations

    def set_k(self, new_k):
        """
        Change the (k) parameter within the class
        """
        self.k = int(new_k)

    def search_movies(self, title=None, year=None):
        """
        Search for movies by title or keyword.
        """
        if title: # a title is provided
            results = self.data[self.data['title'].str.contains(
                title, case=False, na=False)]
            if results.empty:
                print(f'\nNo results found for title: "{title}".')
                return None
            else:
                print(f"\nSearch Results for '{title}':")
                results = results.head(self.k) if len(results) > self.k else results
                for idx, row in results.iterrows():
                    print(f"{row['title']} (IMDb ID: {row['imdbId']})")
            return results
        if year: # a year is provided
            year_val = "(" + year + ")"
            results = self.data[self.data['title'].str.contains(
                year_val, case=False, na=False)]
            if results.empty:
                print(f'\nNo results found for year: "{year}".')
                return None
            else:
                print(f"\nSearch Results for '{year}':")
                results = results.head(self.k) if len(results) > self.k else results
                for idx, row in results.iterrows():
                    print(f"{row['title']} (IMDb ID: {row['imdbId']})")
            return results
        else:
            print(f"\nNo Title or Year was provided...")
            return None

    def clear_search(self):
        """
            Clear search results and show previous recommendations.
        """
        print("\nSearch cleared...")
        if len(self.prev_recommendations) == 0:
            print('No previous recommendations to display')
        else:
            print('Displaying previous recommendations: ')
            for idx, row in self.prev_recommendations.iterrows():
                    print(f"{row['title']} (IMDb ID: {row['imdbId']})")

    def show_pre_model_chart(self, filename=None):
        """
        Generates a 2D chart of the movie feature vectors and optionally saves it as an image file.

        Parameters:
        filename (str): If provided, the chart will be saved to this file.
        """
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(self.features.toarray())

        # Plotting the reduced features
        plt.figure(figsize=(10, 8))
        plt.scatter(reduced_features[:, 0],
                    reduced_features[:, 1], s=10, alpha=0.6)
        plt.title("Movie Feature Vectors (2D Projection)")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")

        for i, title in enumerate(self.data['title'].head(10)):
            plt.text(reduced_features[i, 0],
                     reduced_features[i, 1], title, fontsize=8)

        plt.savefig(filename)
        print(f"Chart saved as {filename}")
        plt.close()

    def show_knn_chart(self, selected_movie_title):
        """
        Visualizes selected movie and its K-nearest neighbors in a 2D chart.
        """
        if not os.path.exists('plots'):
            os.makedirs('plots')

        recommendations = self.recommend(selected_movie_title)

        all_movies = pd.concat(
            [self.data[self.data['title'].isin(self.selected_movies)], recommendations])

        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(self.features.toarray())

        selected_indices = self.data[self.data['title'].isin(
            self.selected_movies)].index
        recommendation_indices = recommendations.index

        plt.figure(figsize=(10, 8))
        plt.scatter(reduced_features[:, 0], reduced_features[:, 1],
                    color='lightgray', s=10, alpha=0.3, label='All Movies')

        plt.scatter(reduced_features[selected_indices, 0], reduced_features[selected_indices, 1],
                    color='red', s=50, label='Selected Movie')

        plt.scatter(reduced_features[recommendation_indices, 0], reduced_features[recommendation_indices, 1],
                    color='blue', s=30, label='Recommended Neighbors')

        for i, title in enumerate(all_movies['title'].head(10)):
            idx = all_movies[all_movies['title'] == title].index[0]
            plt.text(reduced_features[idx, 0],
                     reduced_features[idx, 1], title, fontsize=8)
        # Plot
        plt.title(f"KNN Visualization for '{selected_movie_title}'")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend()
        filename = "plots/KNN_Visualization.png"
        plt.savefig(filename)
        plt.close()


def main():
    # TASK 0: Get dataset from server
    print(f'Task 0: Download dataset from server')
    dataset_file = 'movies_data.csv'
    download_dataset(ICARUS_CS4580_DATASET_URL, dataset_file)
    # TASK 0: Load  data_file into a DataFrame
    print(f'Task 0.0: Load weather data into a DataFrame')
    data_file = f'{DATA_FOLDER}/{dataset_file}'
    data = load_data(data_file)
    print(f'Loaded {len(data)} records')

    movie_engine = MovieRecommendationEngine(data)
    # main loop to interact with the user:
    movie_engine.welcome_message
    while True:
        print(f"\nWhat would you like to do?")
        print(f'1) Get random recommendations')
        print(f'2) Search for a movie by Title or Year')
        print(f'3) Clear search and reset recommendations')
        print(f'4) Get recommendations based on selection? (current k: {movie_engine.k})')
        print(f'5) Change the value of k (current value: {movie_engine.k})')
        print(f"Enter 'q' to quit.")

        main_menu_choice = input("Please choose an option: ").strip().lower()
        if main_menu_choice == 'q':
            print('Ending the program. Goodbye...')
            break

        if main_menu_choice == '1': 
            movie_engine.initial_recommendations()

        elif main_menu_choice == '2':
            repeat_search = 'Y'
            while repeat_search != 'n':
                search_option = input("Would you like to search by Title(t) or Year(y)? Please enter your choice: ").strip().lower()
                if search_option == 'q':
                    print('Ending the program. Goodbye...')
                    break
                elif search_option == 't':
                    year = None
                    user_title = input("What title would you like to search for?: ")
                    movie_engine.search_movies(user_title, year)
                    repeat_search = input("Would you like to search again? (y/n): ").strip().lower()
                elif search_option == 'y':
                    title = None
                    user_year = input("What year would you like to search for?: ")
                    movie_engine.search_movies(title, user_year)
                    repeat_search = input("Would you like to search again? (y/n): ").strip().lower()
                else:
                    repeat_search = input("Invalid entry, would you like to try again? (y/n): ").strip().lower()
        
        elif main_menu_choice == '3':
            movie_engine.clear_search()

        elif main_menu_choice == '4':
            user_title = input("What title would you like to reccomend by?: ")
            movie_engine.recommend(user_title)
            generate_viz = input("Would you like to generate visualizations? (y/n): ").strip().lower()
            if generate_viz == 'y':
                movie_engine.show_knn_chart(user_title)

        elif main_menu_choice == '5':
            retry_k_selection = 'Y'
            while retry_k_selection != 'n':
                k_val = input("What value would you like for k?: ").strip()
                if k_val.isdigit() == False:
                    retry_k_selection = input("Incorrect k-value, would you like to try again?(y/n): ").strip().lower()
                else:
                    movie_engine.set_k(k_val)
                    retry_k_selection = 'n'
            
    #movie_engine.welcome_message()
    #movie_engine.initial_recommendations()
    # movie_engine.recommend()
    #search_key = movie_engine.search_movies("Toy Story")
    #movie_engine.clear_search(search_key)
    # movie_engine.recommend("Toy Story (1995)")
    # movie_engine.recommend("Jumanji (1995)")

    #movie_engine.show_pre_model_chart("plots/prepare_model_chart.png")
    #movie_engine.show_knn_chart(search_key)

if __name__ == '__main__':
    main()
