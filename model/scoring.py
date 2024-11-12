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