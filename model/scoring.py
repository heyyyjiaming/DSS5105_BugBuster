import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import plotly.express as px
    
# def ESG_scoring(esg_data, esg_data_scaled, kmeans, reg):
    # company_info = esg_data[['Sub-sector', 'Company Name', 'Year']]
    # esg_data_numeric = esg_data.drop(columns=['Sub-sector', 'Company Name', 'Year','Recordable work-related ill health cases', 
    #                                           'Average Training Hours per Employee', 'Total Energy Consumption (MWhs)',
    #                                           'Women on the Board (%)', 'Current Employees by Gender (Female %)', 
    #                                           'Women in Management Team (%)', 'Fatalities', 'Board Independence (%)'])
    
    # imputer = SimpleImputer(strategy='median')
    # esg_data_imputed = pd.DataFrame(imputer.fit_transform(esg_data_numeric), columns=esg_data_numeric.columns)
    
    # scaler = StandardScaler()
    # esg_data_scaled = pd.DataFrame(scaler.fit_transform(esg_data_imputed), columns=esg_data_numeric.columns)
    
    # wcss = []  # Within-cluster sum of squares
    # for i in range(1, 11):
    #     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    #     kmeans.fit(esg_data_scaled)
    #     wcss.append(kmeans.inertia_)
        
    # optimal_clusters = 3
    # kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
    # esg_data['Cluster'] = kmeans.predict(esg_data_scaled)
    
    # esg_cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=esg_data_numeric.columns)
    # esg_cluster_centers['Cluster'] = range(optimal_clusters)
    
    # def categorize_performance(cluster):
    #     if cluster == esg_cluster_centers['Cluster'].idxmax():
    #         return 'Good'
    #     elif cluster == esg_cluster_centers['Cluster'].idxmin():
    #         return 'Poor'
    #     else:
    #         return 'Average'

    # esg_data['Performance Category'] = esg_data['Cluster'].apply(categorize_performance)
    
    # X = esg_data_scaled
    # reg = LinearRegression()
    # reg.fit(X, esg_data['Cluster'])
    
    # esg_data['predicted_score'] = reg.predict(X)
    
    # esg_feature_weights = pd.Series(reg.coef_, index=esg_data_numeric.columns).sort_values(ascending=False)
    
    # esg_intercept_b = reg.intercept_
    
    
    # return esg_data, kmeans, esg_cluster_centers, esg_feature_weights, esg_intercept_b



def ESG_trend(esg_data):
    ESG_score_trend = esg_data.groupby('Year')['predicted_score'].mean().reset_index()
    ESG_score_trend.rename(columns = {'predicted_score' : 'Industry mean'}, inplace = True)
    
    soft_serve_esg = esg_data[esg_data['Sub-sector'] == 'Software and Services'].reset_index(drop=True)
    tech_equip_esg = esg_data[esg_data['Sub-sector'] == 'Technology Hardware and Equipment'].reset_index(drop=True)
    semi_esg = esg_data[esg_data['Sub-sector'] == 'Semiconductors and Semiconductor Equipment'].reset_index(drop=True)
    info_tech_esg = esg_data[esg_data['Sub-sector'] == 'Information Technology'].reset_index(drop=True)
    engin_esg = esg_data[esg_data['Sub-sector'] == 'Engineering Services'].reset_index(drop=True)
    
    soft_serve_esg_trend = soft_serve_esg.groupby('Year')['predicted_score'].mean().reset_index()
    soft_serve_esg_trend.rename(columns = {"predicted_score" : "Software and Services"}, inplace = True)

    tech_equip_esg_trend = tech_equip_esg.groupby('Year')['predicted_score'].mean().reset_index()
    tech_equip_esg_trend.rename(columns = {"predicted_score" : "Technology Hardware and Equipment"}, inplace = True)

    semi_esg_trend = semi_esg.groupby('Year')['predicted_score'].mean().reset_index()
    semi_esg_trend.rename(columns = {"predicted_score" : "Semiconductors and Semiconductor Equipment"}, inplace = True)

    info_tech_esg_trend = info_tech_esg.groupby('Year')['predicted_score'].mean().reset_index()
    info_tech_esg_trend.rename(columns = {"predicted_score" : "Information Technology"}, inplace = True)

    engin_esg_trend = engin_esg.groupby('Year')['predicted_score'].mean().reset_index()
    engin_esg_trend.rename(columns = {"predicted_score" : "Engineering Services"}, inplace = True)
    
    sub_sectors_df = ESG_score_trend.merge(soft_serve_esg_trend, on='Year').merge(tech_equip_esg_trend, on='Year').merge(semi_esg_trend, on='Year').merge(info_tech_esg_trend, on='Year').merge(engin_esg_trend, on='Year')
    
    esg_industry_plot_data = sub_sectors_df.melt(id_vars = ["Year"],
                                                 var_name = "sub-sectors", value_name = "predicted_score")
    
    return ESG_score_trend, esg_industry_plot_data
    
    
def ESG_trend_plot(esg_industry_plot_data):
    fig_esg_trend = px.line(esg_industry_plot_data, x = "Year", y = "predicted_score", color = "sub-sectors",
                            markers = True, 
                            title = "Environment score trend of the technology industry and sub-sectors")
    
    fig_esg_trend.update_layout(legend=dict(orientation="h", yanchor="bottom", 
                                            y=-0.5, xanchor="right",x=1)
                                )
    
    fig_esg_trend.update_xaxes(dtick = 1)
    
    fig_esg_trend.update_traces(
        hovertemplate = 'Year: %{x} <br> ESG Score: %{y} <extra></extra>', 
        marker = dict(size = 8)
        )
    
    fig_esg_trend.update_traces(selector = dict(name = 'Industry mean'),
                                line = dict(width = 4, color = "black"), 
                                marker = dict(size = 10, color = "black")) 
    
    return fig_esg_trend




def company_scoring(esg_data, company_data, kmeans, esg_cluster_centers, reg, esg_industry_plot_data, ESG_score_trend):
    sub_sectors = {'Software and Services':['Captii','CSE Global','V2Y Corp','SinoCloud Grp'],
               'Technology Hardware and Equipment':['Addvalue Tech','Nanofilm','Venture'],
               'Semiconductors and Semiconductor Equipment':['AdvancedSystems','AEM SGD','Asia Vets','ASTI','UMS'],
               'Information Technology':['Audience'],
               'Engineering Services':['ST Engineering','Singtel','GSS Energy']}
    
    # esg_data_numeric = esg_data.drop(columns=['Sub-sector', 'Company Name', 'Year', 'GHG Emissions (Total) (tCO2e)', 
    #                                           'Recordable work-related ill health cases', 'Average Training Hours per Employee', 
    #                                           'Total Energy Consumption (MWhs)', 'Women on the Board (%)', 'Current Employees by Gender (Female %)', 
    #                                     'Women in Management Team (%)', 'Fatalities', 'Board Independence (%)']).reset_index(drop=True)
    
    company_data = company_data[company_data['Year'].between(2020, 2024)]
    company_info = company_data[['Company Name', 'Year']]
    company_name = str(company_data.iloc[0]['Company Name'])
    company_numeric = company_data.drop(columns=['Company Name', 'Year','Recordable work-related ill health cases', 'GHG Emissions (Total) (tCO2e)',
                                                 'Average Training Hours per Employee', 'Total Energy Consumption (MWhs)',
                                                 'Women on the Board (%)', 'Current Employees by Gender (Female %)', 
                                                 'Women in Management Team (%)', 'Fatalities', 'Board Independence (%)'])
    
    # Handle missing values(Implement techniques to handle missing data and ensure fair comparisons across companies.)
    imputer = SimpleImputer(strategy='median')
    company_data = pd.DataFrame(imputer.fit_transform(company_numeric), columns=company_numeric.columns)
    
    # Standardize Data
    scaler = StandardScaler()
    company_scaled = pd.DataFrame(scaler.fit_transform(company_data), columns=company_numeric.columns)
    
    # Use KMeans to set Performance Category
    # Use trained KMeans on input company
    company_scaled_data = company_scaled.values  
    
    
    # company_clusters = kmeans.predict(company_scaled_data)
    # company_data['Cluster'] = company_clusters
    
    # # Set Performance Category based on clusters
    # def categorize_performance_by_cluster(cluster):
    #     if cluster == esg_cluster_centers['Cluster'].idxmax():
    #         return 'Good'
    #     elif cluster == esg_cluster_centers['Cluster'].idxmin():
    #         return 'Poor'
    #     else:
    #         return 'Average'
    
    # company_data['Performance Category'] = company_data['Cluster'].apply(categorize_performance_by_cluster)
    
    # # Calculate ESG Score for new company
    # # res=company_scaled.columns
    # company_scores = reg.predict(company_scaled)
    # # def calculate_score(features, weights, intercept):
    # #     return np.dot(features, weights) + intercept
    # # esg_feature_weights = pd.Series(reg.coef_, index=esg_data_numeric.columns).sort_values(ascending=False)
    # # esg_intercept_b = reg.intercept_
    # # esg_weights = np.array(esg_feature_weights)
    # # company_scores = company_scaled.apply(lambda row: calculate_score(row, esg_weights, esg_intercept_b), axis=1)
    
    # company_data['Calculated Score'] = company_scores
    # company_data = pd.concat([company_info.reset_index(drop=True), company_data], axis=1)
    
    # company_score = company_data[['Year', 'Calculated Score']]
    # company_score.rename(columns = {'Calculated Score': company_name}, inplace = True)
    
    # # Loop through each company and check if target_value is in its list of industries using isin
    # for sub_sector in sub_sectors:
    #     if pd.Series(sub_sectors[sub_sector]).isin([company_name]).any():
    #         company_sub_sector = sub_sector
    
    # sub_sector_select = esg_industry_plot_data[esg_industry_plot_data["sub-sectors"] == company_sub_sector]
    # sub_sector_select = sub_sector_select.drop(columns = {'sub-sectors'})
    # sub_sector_select.rename(columns = {'predicted_score':company_sub_sector}, inplace = True)
    
    # compare_data = ESG_score_trend.merge(company_score, on = 'Year').merge(sub_sector_select, on = 'Year')
    # compare_data = compare_data.melt(id_vars = ["Year"],
    #                                  var_name = "Type", value_name = "predicted_score")
    
    # fig_compare = px.line(compare_data, x = "Year", y = "predicted_score", color = "Type",
    #                       markers = True, title = "Comparison on ESG score trend")
    
    # fig_compare.update_traces(
    #     hovertemplate = 'Year: %{x} <br> ESG Score: %{y} <extra></extra>', 
    #     marker = dict(size = 8)
    #     )
  
    # return fig_compare
    return company_scaled.columns, reg.feature_names_in_
    