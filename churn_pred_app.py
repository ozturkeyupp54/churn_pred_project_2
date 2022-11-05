import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image 
import streamlit.components.v1 as components
import codecs
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import sweetviz as sv
import os
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import tensorflow as tf
from keras import Model
from keras.models import load_model



import streamlit as st

from tempfile import NamedTemporaryFile

st.set_option('deprecation.showfileUploaderEncoding', False)

buffer = st.file_uploader("Image here pl0x")
temp_file = NamedTemporaryFile(delete=False)

def st_display_sweetviz(report_html,width=1000,height=500):
        report_file = codecs.open(report_html,'r')
        page = report_file.read()
        components.html(page,width=width,height=height,scrolling=True)
        
# today=st.date_input("Today is", datetime.datetime.now())
def explore_data(dataset):
        df = pd.read_csv(os.path.join(dataset))
        return df 
# st.set_page_config(layout="wide")
# def cat_list(df):
#     cat_list = ["departments", "salary"]
#     index = 0
#     plt.figure(figsize=(16, 12))
#     for i in cat_list:
#         index += 1
#         plt.subplot(2, 2, index)
#         sns.countplot(data=df, x=i, hue="left")
def main():
    """A Simple EDA App with Streamlit Components"""

    menu = ["EDA","Profile Report","Sweetviz","K-Means",'KNN',"Random Forest Model","XGBClassifier","ANN Model"]
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Profile Report":
            st.subheader("Automated EDA with Pandas Profile")
            data_file = 'profile_report.csv'
            if data_file is not None:
                df = pd.read_csv(data_file)
                st.dataframe(df.head())
                profile = ProfileReport(df)
                st_profile_report(profile)

    elif choice == "Sweetviz":
        st.subheader("Automated EDA with Sweetviz")
        # data_file = st.file_uploader("Upload CSV",type=['csv'])
        data_file = 'profile_report.csv'
        if data_file is not None:
            df = pd.read_csv(data_file)
            st.dataframe(df.head())
            if st.button("Generate Sweetviz Report"):

                # Normal Workflow
                report = sv.analyze([df, "original data"],target_feat='left')
                report.show_html(open_browser= True)
                st_display_sweetviz("SWEETVIZ_REPORT.html")


    elif choice == "Random Forest Model":
        st.subheader("Random Forest Classifier")
        st.image('image_37.PNG')

        if st.checkbox("Train | Test Split"):
                st.write("test_size=0.2, random_state=42")
                
        if st.checkbox("Encoding"):
                st.write("'departments', 'salary'")
                st.write("Ordinal Encoding ")
                
        if st.checkbox("Model"):
                st.write("Pipeline Model")
                
        if st.checkbox("Model Performance"):
                st.write("Eval Metric Function")
                
        if st.checkbox("Random Forest"):
                st.write("Pipe Model Eval Metric")
                st.image('image_38.PNG')

        if st.checkbox("Cross Validate"):
                st.image('image_39.PNG')

        if st.checkbox("Evaluating ROC Curves and AUC"):
                image40 = Image.open('image_40.png')
                st.image(image40)

        if st.checkbox("RF Model Feature Importance"):
                st.image('image_41.PNG')
                image42 = Image.open('image_42.png')
                st.image(image42)

        if st.checkbox("Find Best Parameters"):
                st.image('image_43.PNG')

        if st.checkbox("Grid Model Best Parameters"):
                    st.image('image_44.PNG')

        if st.checkbox("Model Perfomance With Grapichs"):
                    st.write("Recall Curve")
                    image45 = Image.open('image_45.png')
                    st.image(image45)
                    st.write('ROC Curve')
#                     st.image('image_46.PNG')
                    image46 = Image.open('image_46.png')
                    st.image(image46)
        if st.checkbox("Feature İmportance"):
                    st.image('image_47.jpg')

        if st.checkbox("Final Model"):
                    st.write('Pipe Model Grid With Five Features')
                    st.image('image_48.PNG')

        if st.checkbox("Save and Export the Model as .pkl and Prediction"):
                    
                    st.image('image_49.PNG')
                    st.write('Prediction')
                    st.image('image_50.PNG')


        if st.checkbox("Final Prediction for Random Forest"):
            filename = 'random_pickle_model'
            model = pickle.load(open(filename, 'rb'))
            # scaled_random= pickle.load(open("random_pipeline","rb"))

            st.sidebar.title("Final Model ")
            # st.sidebar.header("Sidebar header")
            sl=st.sidebar.slider(label='satisfaction_level',min_value=0.0,max_value=1.0,step=0.01,)
            le=st.sidebar.slider(label="last_evaluation:",min_value=0.0,max_value=1.0,step=0.01,)
            nump=st.sidebar.slider("number_project:",min_value=1,max_value=10,step=1,)
            amh=st.sidebar.slider("average_monthly_hours:",min_value=0,max_value=320,step=1,)
            tsc=st.sidebar.slider("time_spend_company:",min_value=0,max_value=12,step=1,)

            dict={"satisfaction_level":sl,
                "last_evaluation":le,
                "number_project":nump,
                "average_montly_hours":amh,
                "time_spend_company":tsc}

            df= pd.DataFrame.from_dict([dict])

            st.table(df)

            if st.button("Predict_Random"):
                predictions = model.predict(df)

                df["pred"] = predictions

                st.write(predictions[0])


    elif choice == "K-Means":
        st.write('# K-MEANS')
        st.image('image_28.PNG')
       
        if st.checkbox("Hopkins Score"):
                st.write("0.14908358236837996")

        if st.checkbox("Elbow Method"):
               image29 = Image.open('image_29.png')
               st.image(image29)

        if st.checkbox("X_diff"):
                st.image("image_30.PNG")

        if st.checkbox("Yellow Brick"):
                image31 = Image.open('image_31.png')
                st.image(image31)

        if st.checkbox("silhouette_score"):
                st.image("image_32.PNG")

        if st.checkbox("silhouette_visualizer"):
                image33 = Image.open('image_33.png')
                st.image(image33)
                
        if st.checkbox("predicted_clusters"):
                st.image("image_34.PNG")
                
        if st.checkbox("Distribution of Clusters"):
                image35 = Image.open('image_35.png')
                st.image(image35)
                
        if st.checkbox("Centroits of Clusters"):
                image36 = Image.open('image_36.png')
                st.image(image36)

    elif choice == "XGBClassifier":
        st.write('# XGBClassifier')
        st.write('# Vanilla Model')


        if st.checkbox("Vanilla Result"):
                st.image("image_59.PNG")

        if st.checkbox("XGBoost Cross Validation"):

                  st.image("image_61.PNG")

        if st.checkbox("XGBoost Gridsearch"):
                st.image("image_63.PNG")

        if st.checkbox("Gridsearch Result"):
                st.image("image_62.PNG")

        if st.checkbox("AUC"):
                image64 = Image.open('image_64.png')
                st.image(image64)
                
        if st.checkbox("Grid Search Feature Importance"):
                st.image("image_65.PNG")
                
        if st.checkbox("New X_train"):
                st.image("image_66.PNG")
                
        if st.checkbox("New Result"):
                st.image("image_67.PNG")
                

        if st.checkbox("Final Prediction for XGBoost"):
            xgboost_m = 'pipe_model_pkl222'
#             xg_model = pickle.load(open(xgboost_m, "rb"))
            xg_model = pickle.load(open('pipe_model_pkl222', "rb"))
            # scaled_random= pickle.load(open("random_pipeline","rb"))

            st.sidebar.title("XGBOOST Model")
            # st.sidebar.header("Sidebar header")
            sl=st.sidebar.slider(label='satisfaction_level',min_value=0.0,max_value=1.0,step=0.01,)
            le=st.sidebar.slider(label="last_evaluation:",min_value=0.0,max_value=1.0,step=0.01,)
            nump=st.sidebar.slider("number_project:",min_value=1,max_value=10,step=1,)
            amh=st.sidebar.slider("average_monthly_hours:",min_value=0,max_value=320,step=1,)
            tsc=st.sidebar.slider("time_spend_company:",min_value=0,max_value=12,step=1,)

            dict={"satisfaction_level":sl,
                "last_evaluation":le,
                "number_project":nump,
                "average_montly_hours":amh,
                "time_spend_company":tsc}

            df_xg= pd.DataFrame.from_dict([dict])

            st.table(df_xg)

            if st.button("Predict_xg"):
                predictions = xg_model.predict(df_xg)

                df_xg["pred"] = predictions

                st.write(predictions[0])


    elif choice == "KNN":
        st.write('# KNN model building')
        st.write('# Train test split yaptık.\n Ardından pipeline oluşturmak adına OneHotEncoder,OrdinalEncoder ve StandardScaler algoritmalarını kullandık')
        st.image('image_68.PNG')
       
        if st.checkbox("KNN vanilla eval metric"):
                st.image("image_69.PNG")
       
        if st.checkbox("KNN elbow"):
            st.write("30 komşuluğuktaki hata durumunu gösteriyor")
            image70 = Image.open('image_70.png')
            st.image(image70)
 
        if st.checkbox("KNN n_neighbors = 2"):
            st.image("image_71.PNG")
 
        if st.checkbox("KNN n_neighbors = 3"):
            st.image("image_72.PNG")
 
        if st.checkbox("KNN n_neighbors = 5"):
            st.image("image_73.PNG")
 
        if st.checkbox("KNN n_neighbors = 10"):
            st.image("image_74.PNG")
 
        if st.checkbox("KNN Cross_Validation"):
            st.image("image_75.PNG")
            st.image("image_76.PNG")


        if st.checkbox("Final Prediction for KNN"):
            KNN_m = 'knn_pipe_model'
            KNN_model = pickle.load(open(KNN_m, 'rb'))
            # scaled_random= pickle.load(open("random_pipeline","rb"))

            st.sidebar.title("KNN Model ")
            # st.sidebar.header("Sidebar header")
            sl=st.sidebar.slider(label='satisfaction_level',min_value=0.0,max_value=1.0,step=0.01,)
            le=st.sidebar.slider(label="last_evaluation:",min_value=0.0,max_value=1.0,step=0.01,)
            nump=st.sidebar.slider("number_project:",min_value=1,max_value=10,step=1,)
            amh=st.sidebar.slider("average_monthly_hours:",min_value=0,max_value=320,step=1,)
            tsc=st.sidebar.slider("time_spend_company:",min_value=0,max_value=12,step=1,)

            wa=st.sidebar.slider("work_accident:",min_value=0,max_value=1,step=1,)
            
            plS=st.sidebar.slider("promotion_last_5years:",min_value=0,max_value=1,step=1,)

            dpr=st.sidebar.selectbox("Select a department", ['sales', 'accounting', 'hr', 'technical', 'support', 'management','IT', 'product_mng', 'marketing', 'RandD'])
            
            # slry=st.sidebar.slider("salary:",min_value=0,max_value=1,step=1,)
            slry=st.sidebar.selectbox("Select a salary type",["low","medium","high"])


            dict={"satisfaction_level":sl,
                "last_evaluation":le,
                "number_project":nump,
                "average_montly_hours":amh,
                "time_spend_company":tsc,
                "work_accident":wa,

                "promotion_last_5years":plS,
                "departments":dpr,
                "salary":slry}

            df= pd.DataFrame.from_dict([dict])

            st.table(df)

            if st.button("Predict_KNN"):
                predictions = KNN_model.predict(df)

                df["pred"] = predictions

                st.write(predictions[0])


    elif choice == "ANN Model":
        st.write('# ANN Model')
        # st.image('kmeansmeme.PNG')
       
        if st.checkbox("Train-Test Split and Shape"):
                st.write("Test_size = 0.10")
                st.image("image_52.PNG")
       
        if st.checkbox("Get Paramaters and Model Fit"):
 
                st.image("image_53.PNG")
       
        if st.checkbox("Evaluating Model Performance and Tunning"):
     
                image54 = Image.open('image_54.png')
                st.image(image54)

                st.image("image_55.PNG")

        if st.checkbox("ANN Confuion Matrix and Classification Report "):
 
                st.image("image_56.PNG")


        if st.checkbox("Final Prediction for ANN"):
            filename = 'model_churn'
            model = load_model(filename)
            

            st.sidebar.title("ANN Model ")
            # st.sidebar.header("Sidebar header")
            
            sl=st.sidebar.slider(label='satisfaction_level',min_value=0.0,max_value=1.0,step=0.01,)
            
            le=st.sidebar.slider(label="last_evaluation:",min_value=0.0,max_value=1.0,step=0.01,)
            
            nump=st.sidebar.slider("number_project:",min_value=1,max_value=10,step=1,)
            
            amh=st.sidebar.slider("average_monthly_hours:",min_value=0,max_value=320,step=1,)
            
            tsc=st.sidebar.slider("time_spend_company:",min_value=0,max_value=12,step=1,)
            
            slry=st.sidebar.slider("salary:",min_value=0,max_value=2,step=1,)
            
            wa=st.sidebar.slider("Work_accident:",min_value=0,max_value=1,step=1,)
            
            plS=st.sidebar.slider("promotion_last_5years:",min_value=0,max_value=1,step=1,)

            dict={"satisfaction_level":sl,
                "last_evaluation":le,
                "number_project":nump,
                "average_montly_hours":amh,
                "time_spend_company":tsc,
                "salary":slry,
                "Work_accident":wa,
                "promotion_last_5years":plS}

            df= pd.DataFrame.from_dict([dict])
            # sample_scaled = scaled_ann.transform(df)
            # st.table(df)

            if st.button("Predict_ANN"):
                predictions = model.predict(df)

                df["pred"] = predictions

                st.write(predictions[0])










    else:
            st.subheader("EDA")
            st.write("## Zemin güzel, hava güzel, tahmin yürütmek için her şey müsait")

            # st.code("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nfrom sklearn.metrics import classification_report\nconfusion_matrix\nfrom sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score\nfrom sklearn.model_selection import cross_val_score, cross_validate\nfrom sklearn.model_selection import GridSearchCV")
            # st.code("import tensorflow as tf\nfrom keras.models import Sequential\nfrom keras.layers import Dense, Activation, Dropout\nfrom keras.callbacks import EarlyStopping\nfrom keras.optimizers import Adam")

            st.write("# HR - Employee Churn Prediction")
            st.image("image_1.jpg")
            st.write("# Dataframes")


            my_dataset = "profile_report.csv"
            # Load Our Dataset
            data = explore_data(my_dataset)

            # Show Dataset
            if st.checkbox("Preview DataFrame"):
                if st.button("Head"):
                    st.write(data.head())
                if st.button("Tail"):
                    st.write(data.tail())
                else:
                    st.write(data.head(2))

            # Show Entire Dataframe
            if st.checkbox("Show All DataFrame"):
                st.dataframe(data)

            # Show All Column Names
            if st.checkbox("Show All Column Name"):
                st.text("Columns:")
                st.write(data.columns)


            # Show Summary of Dataset
            if st.checkbox("Show İnfo of Dataset"):
#                 image2 = Image.open('image_2.png')
#                 st.image(image2)
                st.image("image_2.PNG")
                
            if st.checkbox("Show Summary of Dataset"):
                st.write(data.describe().T)

            if st.checkbox("Show mean all columns"):
#                 image3 = Image.open('image_3.png')
#                 st.image(image3)
                  st.image("image_3.PNG")
                

            if st.checkbox("num_pro_val_count"):
#                 image4 = Image.open('image_4.png')
#                 st.image(image4)
                  st.image("image_4.PNG")

            if st.checkbox("time_spend_value_normalize"):
#                 image5 = Image.open('image_5.png')
#                 st.image(image5)
                  st.image("image_5.PNG")

            if st.checkbox("barh_plot"):
                image6 = Image.open('image_6.png')
                st.image(image6)

            if st.checkbox("histogram"):
                image7 = Image.open('image_7.png')
                st.image(image7)
            
            if st.checkbox("cat plot"):
                image8 = Image.open('image_8.png')
                st.image(image8)
            
            if st.checkbox("employees left"):
                image9 = Image.open('image_9.png')
                st.image(image9)
            
            if st.checkbox("Number of Projects"):
                image10 = Image.open('image_10.png')
                st.image(image10)
            
            if st.checkbox("Time Spent in Company"):
                image11 = Image.open('image_11.png')
                st.image(image11)

            if st.checkbox("The number of projects and the number of lefts"):
                image12 = Image.open('image_12.png')
                st.image(image12)
            
            if st.checkbox("Subplots time_spend_company and lefts"):
                image57 = Image.open('image_57.png')
                st.image(image57)
            
            if st.checkbox("Value count and percentage of time spent company"):
                image14 = Image.open('image_14.png')
                st.image(image14)
            
            if st.checkbox("number_project_left"):
                image15 = Image.open('image_15.png')
                st.image(image15)
            
            if st.checkbox("satisfaction_level"):
                image16 = Image.open('image_16.png')
                st.image(image16)
            
            if st.checkbox("average_montly_hours"):
                image17 = Image.open('image_17.png')
                st.image(image17)
            
            if st.checkbox("Promotion last 5 years"):
                image18 = Image.open('image_18.png')
                st.image(image18)

            if st.checkbox("Number of Lefts by Department"): # bunu da 
                image19 = Image.open('image_19.png')
                st.image(image19)
            
            if st.checkbox("Percentage of Lefts by Department"):
                image20 = Image.open('image_20.png')
                st.image(image20)
            
            if st.checkbox("Number of employees who left their jobs by salary status"):
                image21 = Image.open('image_21.png')
                st.image(image21)
            
            if st.checkbox("Correlation table and heatmap"):
                image22 = Image.open('image_22.png')
                st.image(image22)

            if st.checkbox("satisfaction_level_and_last_evaluation"):
                image23 = Image.open('image_23.png')
                st.image(image23)
 
            if st.checkbox("satisfaction_level_and_average_monthly_hours"):
                image24 = Image.open('image_24.png')
                st.image(image24)
 
            if st.checkbox("satisfaction_level_and_salary"):
                image25 = Image.open('image_25.png')
                st.image(image25)
 
            if st.checkbox("satisfaction_level_and_promotion_last_5years"):
                image27 = Image.open('image_27.png')
                st.image(image27)

          

if __name__ == '__main__':
        main()



