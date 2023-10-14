import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from kmodes.kmodes import KModes
from sklearn.feature_selection import RFE
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import PowerTransformer, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.decomposition import PCA
from supabase import Client, create_client
from sentence_transformers import SentenceTransformer
import os
import json
import logging
from dotenv import load_dotenv, find_dotenv
import openai
from datetime import datetime
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

load_dotenv()


logger = logging.getLogger(__name__)
log_file_path = f'/Users/kisazehra/Documents/solarplexus/segmentation/app/logs/segmentation_{timestamp}.log'
file_handler = logging.FileHandler(log_file_path)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)



OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

openai.api_key = OPENAI_API_KEY
AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase_bucket = 'solarplexus'



try:

    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    logger.info('Supabase connection successfull')

except Exception as e:
    print(f"Error connecting to Supabase: {e}")
    logger.error(e)


 
def segmentation(project_id, user_id, parameters, feature_selection):

    print("entered")
    try:
        process_data_insert = [
                    {
                        "user_id" :user_id,
                        "process_type": "data_segmentation",
                        "process_status": "in_progress",
                        "start_at" : datetime.now().isoformat()
                    },
                ]


        process= supabase.from_("process").insert(process_data_insert).execute()
        process_data = [record for record in process.data]

        p_data = json.dumps(process_data)
        p = json.loads(p_data)
        process_id = p[0]["process_id"]

        print("step 11111111111")
        file = supabase.from_("Project_files").select("*").eq("project_id",project_id).execute()

        file_data = [record for record in file.data]
        print(file_data)
        data = json.dumps(file_data)
        d = json.loads(data)

        file_path = d[0]["file_path"]




        print("step 222222222222222222222222")
        try:

            local_file_path = f'/Users/kisazehra/Documents/solarplexus/segmentation/app/target_files/{file_path.split("/")[-1]}'
            print(local_file_path)
            # s3_folder.download_file(Key=file_path, Filename=local_file_path)
            print("step 333333333333333333333333")

            print(file_path)
            with open(local_file_path, 'wb+') as f:
                data = supabase.storage.from_(supabase_bucket).download(file_path)
                f.write(data)

            print("step 444444444444444444")

        except Exception as e:

            logging.error('An error occurred:', exc_info=True)




        df = pd.read_excel(local_file_path)

        print("step 55555555555555555555")
        print("df shape----------------------",df.shape)

        df.dropna(axis=1, how='all', inplace=True)

        print("step 6666666666666666666")
        print("dropped null shape----------------------",df.shape)

        for column in df.columns:
            if df[column].dtype == 'object':
                # Impute with mode for string columns
                df[column].fillna(df[column].mode()[0], inplace=True)
            else:
                # Impute with mean for numeric columns
                df[column].fillna(df[column].mean(), inplace=True)






        # Apply the dynamic compile_text function to the DataFrame
        sentences = df.to_dict(orient='records')  # Convert DataFrame to a list of dictionaries
        formatted_texts = [compile_text(data) for data in sentences]

        # # Example usage:
        # for text in formatted_texts:
        #     print("-------")
        #     print(text)


        # -------------------- Second Step --------------------

        model = SentenceTransformer(r"sentence-transformers/paraphrase-MiniLM-L6-v2")
        output = model.encode(sentences=formatted_texts,
                show_progress_bar=True,
                normalize_embeddings=True)

        df_embedding = pd.DataFrame(output)
        print(df_embedding.head(5))






        """

                # Preprocessing part


        numeric_data = df.select_dtypes(include=['int64', 'float64'])
        categorical_data = df.select_dtypes(include=['object']) 
        categorical_columns = categorical_data.columns
 








        print("categorical columnss-----",categorical_columns.tolist())
        print(categorical_data.info())
        print("numeric columnss-----",numeric_data.columns.tolist())
        print(numeric_data.info())

        # Feature scaling (standardization)
        scaler = StandardScaler()
        numeric_data_scaled = scaler.fit_transform(numeric_data)
        # print(numeric_data_scaled.columns)
        # Perform PCA for dimensionality reduction
        # pca = PCA(n_components=0.95)
        # numeric_data_pca = pca.fit_transform(numeric_data_scaled)

        # Merge the scaled numeric data with the categorical data
        combined_data = pd.concat([pd.DataFrame(numeric_data_scaled), categorical_data], axis=1)
        print(combined_data.shape)
        print(combined_data.columns.tolist())
        print(combined_data.info())

        # Initialize variables for best silhouette score and corresponding cluster count
        best_silhouette_score = -1
        optimal_num_clusters = 2  # Initial value, will be updated

        # Perform clustering with different cluster counts and calculate silhouette score
        for num_clusters in range(2, len(combined_data)):  # Adjust the range as needed
            print("-----------")
            print(num_clusters)

            kproto = KPrototypes(n_clusters=num_clusters, init='Cao', verbose=0)
            # k_modes = KModes(n_clusters=num_clusters)
            print("===========")
            # Fit the k-modes model to your data
            # clusters = k_modes.fit_predict(combined_data)

            

            # categorical = [col in categorical_columns for col in combined_data.columns]
            clusters = kproto.fit_predict(combined_data, categorical=combined_data.columns.tolist())

            # clusters = kproto.fit_predict(combined_data, categorical=[categorical_columns])
            print("*********")
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(combined_data, clusters)
            print(":::::::::::::::::")
            
            # Update optimal number of clusters if a higher silhouette score is found
            if silhouette_avg > best_silhouette_score:
                print("iffffffffff")
                best_silhouette_score = silhouette_avg
                print(".................")
                optimal_num_clusters = num_clusters
                print(optimal_num_clusters)
            print(num_clusters)  


        print("finalized optimal_num_clusters-----", optimal_num_clusters)
        # Re-fit the model with the optimal number of clusters
        # kproto = KPrototypes(n_clusters=optimal_num_clusters, init='Cao', verbose=0)
        # clusters = kproto.fit_predict(combined_data, categorical=[categorical_columns])

        # Add cluster labels to the original dataset
        # data['Cluster'] = clusters

        df.loc[:, 'Cluster'] = clusters

        print("step 144444444444444444")
        # Calculate the Silhouette Score for your clustering result
        silhouette_avg_kmeans = silhouette_score(df, clusters)
        print("Silhouette Score K-Means:", silhouette_avg_kmeans)

        print("step 15555555555555555555")



        df.to_csv(f"cluster_{user_id}_{project_id}.csv")"""

        return "segment generated successfully"



    except Exception as e:
        logger.error(e)
        print(e)
        return {"error": e}


# -------------------- Dynamic compile_text Function --------------------
def compile_text(data_dict):
    # Initialize an empty list to store key-value pairs
    key_value_pairs = []

    # Iterate through the dictionary or iterable and format each key-value pair
    for key, value in data_dict.items():
        key_value_pairs.append(f"{key}: {value}")

    # Join the key-value pairs with line breaks to create the final string
    formatted_text = "\n".join(key_value_pairs)

    return formatted_text



# segmentation(1,"1", {}, 'yes')
