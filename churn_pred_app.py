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

def st_display_sweetviz(report_html,width=1000,height=500):
        report_file = codecs.open(report_html,'r')
        page = report_file.read()
        components.html(page,width=width,height=height,scrolling=True)
        
# today=st.date_input("Today is", datetime.datetime.now())
def explore_data(dataset):
        df = pd.read_csv(os.path.join(dataset))
        return df 

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

    menu = ["EDA","Profile Report","Sweetviz",'Final Model',"K-Means",'KNN',"Random Forest Model","ANN Model"]
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
                report.show_html()
                st_display_sweetviz("SWEETVIZ_REPORT.html")


    elif choice == "Random Forest Model":
        st.subheader("Random Forest Classifier")
        st.image('random_tree.PNG')

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
                st.image('random_forest_eval_metric.PNG')

        if st.checkbox("Cross Validate"):
                st.image('random_cv1.PNG')

        if st.checkbox("Evaluating ROC Curves and AUC"):
                st.image('random_roc_curve.PNG')

        if st.checkbox("RF Model Feature Importance"):
                st.image('random_fut_imp2.PNG')
                st.image('random_fut_imp.PNG')

        if st.checkbox("Find Best Parameters"):
                st.image('random_best_param.PNG')

        if st.checkbox("Grid Model Best Parameters"):
                    st.image('random_grid_best_eval_param.PNG')

        if st.checkbox("Model Perfomance With Grapichs"):
                    st.write("Recall Curve")
                    st.image('random_recall_curve.PNG')
                    st.write('ROC Curve')
                    st.image('random_ROC_curve1.PNG')

        if st.checkbox("Feature İmportance"):
                    st.image('feature_importance.jpg')

        if st.checkbox("Final Model"):
                    st.write('Pipe Model Grid With Five Features')
                    st.image('random_five_features_eval.PNG')

        if st.checkbox("Save and Export the Model as .pkl and Prediction"):
                    
                    st.image('random_final_pickle_code.PNG')
                    st.write('Prediction')
                    st.image('random_prediction_1.PNG')


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

            if st.button("Predict"):
                predictions = model.predict(df)

                df["pred"] = predictions

                st.write(predictions[0])


    elif choice == "K-Means":
        st.write('# K-MEANS')
        st.image('kmeansmeme.PNG')
       
        if st.checkbox("Hopkins Score"):
                st.write("0.14908358236837996")

        if st.checkbox("Elbow Method"):
                st.image("elbow_method.PNG")

        if st.checkbox("X_diff"):
                st.image("x_diff.PNG")

        if st.checkbox("Yellow Brick"):
                st.image("yellow_brick.PNG")

        if st.checkbox("silhouette_score"):
                st.image("silüet_skor.PNG")

        if st.checkbox("silhouette_visualizer"):
                st.image("silhouette_visualizer.PNG")
                
        if st.checkbox("predicted_clusters"):
                st.image("predicted_clusters.PNG")
                
        if st.checkbox("Distribution of Clusters"):
                st.image("dist_clusters.PNG")
                
        if st.checkbox("Centroits of Clusters"):
                st.image("clusters_centers.PNG")

    elif choice == "KNN":
        st.write('# K-MEANS')
        st.image('kmeansmeme.PNG')
       
        if st.checkbox("Hopkins Score"):
                st.write("0.14908358236837996")

        if st.checkbox("Elbow Method"):
                st.image("elbow_method.PNG")

        if st.checkbox("X_diff"):
                st.image("x_diff.PNG")

        if st.checkbox("Yellow Brick"):
                st.image("yellow_brick.PNG")

        if st.checkbox("silhouette_score"):
                st.image("silüet_skor.PNG")

        if st.checkbox("silhouette_visualizer"):
                st.image("silhouette_visualizer.PNG")
                
        if st.checkbox("predicted_clusters"):
                st.image("predicted_clusters.PNG")
                
        if st.checkbox("Distribution of Clusters"):
                st.image("dist_clusters.PNG")
                
        if st.checkbox("Centroits of Clusters"):
                st.image("clusters_centers.PNG")

    elif choice == "ANN Model":
        st.write('# ANN Model')
        # st.image('kmeansmeme.PNG')
       
        if st.checkbox("Train-Test Split and Shape"):
                st.write("Test_size = 0.10")
                st.image("ann_shape1.PNG")
       
        if st.checkbox("Get Paramaters and Model Fit"):
 
                st.image("ann_param_fit.PNG")
       
        if st.checkbox("Evaluating Model Performance and Tunning"):
     
                st.image("ann_model_history.PNG")
                st.image("ann_vall_loss.PNG")

        if st.checkbox("ANN Confuion Matrix and Classification Report "):
 
                st.image("ann_f1_score1.PNG")










    else:
            st.subheader("EDA")
            st.write("## Zemin güzel, hava güzel, tahmin yürütmek için her şey müsait")

            # st.code("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport warnings\nfrom sklearn.metrics import classification_report\nconfusion_matrix\nfrom sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score\nfrom sklearn.model_selection import cross_val_score, cross_validate\nfrom sklearn.model_selection import GridSearchCV")
            # st.code("import tensorflow as tf\nfrom keras.models import Sequential\nfrom keras.layers import Dense, Activation, Dropout\nfrom keras.callbacks import EarlyStopping\nfrom keras.optimizers import Adam")

            st.write("# HR - Employee Churn Prediction")
            st.image("eva.jpg")
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
                st.image("info.PNG")

            if st.checkbox("Show Summary of Dataset"):
                st.write(data.describe().T)

            if st.checkbox("Show mean all columns"):
                st.image("left_mean_all.PNG")

            if st.checkbox("num_pro_val_count"):
                st.image("num_pro_val_count.PNG")

            if st.checkbox("time_spend_value_normalize"):
                st.image("tim_spren_val_norm.PNG")

            if st.checkbox("barh_plot"):
                st.image("barh_plot.PNG")


            if st.checkbox("histogram"):
                st.image("histogram.PNG")
            
            if st.checkbox("cat plot"):
                st.image("cat_plot.PNG")
            
            if st.checkbox("employees left"):
                st.image("employees_left.PNG")
            
            if st.checkbox("Number of Projects"):
                st.image("NoP.PNG")
            
            if st.checkbox("Time Spent in Company"):
                st.image("tsic.PNG")

            if st.checkbox("The number of projects and the number of lefts"):
                st.image("subnumpro.PNG")
            
            if st.checkbox("Subplots time_spend_company and lefts"):
                st.image("timespendcomp.PNG")
            
            if st.checkbox("Value count and percentage of time spent company"):
                st.image("time_spend_comp_left.PNG")
            
            if st.checkbox("number_project_left"):
                st.image("number_project_left.PNG")
            
            if st.checkbox("satisfaction_level"):
                st.image("satisfaction_level.PNG")
            
            if st.checkbox("average_montly_hours"):
                st.image("average_montly_hours.PNG")
            
            if st.checkbox("Promotion last 5 years"):
                st.image("promotion_last_5years.PNG")

            if st.checkbox("Number of Lefts by Department"): # bunu da 
                st.image("departments_left.PNG")
            
            if st.checkbox("Percentage of Lefts by Department"):
                st.image("departments_left_2.PNG")
            
            if st.checkbox("Number of employees who left their jobs by salary status"):
                st.image("salary_left.PNG")
            
            if st.checkbox("Correlation table and heatmap"):
                st.image("corr_heatmap.PNG")

            if st.checkbox("satisfaction_level_and_last_evaluation"):
                st.image("satisfaction_level_and_last_evaluation.PNG")
 
            if st.checkbox("satisfaction_level_and_average_monthly_hours"):
                st.image("satisfaction_level_and_average_monthly_hours.PNG")
 
            if st.checkbox("satisfaction_level_and_salary"):
                st.image("satisfaction_level_and_salary.PNG")
 
            if st.checkbox("satisfaction_level_and_promotion_last_5years"):
                st.image("satisfaction_level_and_promotion.PNG")

          

if __name__ == '__main__':
        main()




