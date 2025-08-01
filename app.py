import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None

# Dashboard title
st.title('Zidisha Loans Monthly Analyser')

# Combined format notice and file upload in one expander
with st.expander("📁 Upload CSV File", expanded=False):
    st.write("""
    **Expected CSV Format:**
    Your CSV file should contain the following columns:
    - Branch Name
    - Client Name  
    - Disbursed On Date
    - Loan Officer Name
    - Principal Amount
    - Expected Repayment
    - Total Outstanding
    - Total Repayment
    
    *Note: Column names can have extra spaces and will be automatically cleaned.*
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Save the uploaded file as "data.csv"
        with open("data.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File uploaded successfully!")
        
        # Reset session state to reload data
        st.session_state.data_loaded = False
        st.session_state.df = None
        
        # Add refresh button
        if st.button("Refresh Analytics"):
            st.rerun()

# Load data
def load_data():
    # Check if the file exists
    if not os.path.exists('data.csv'):
        st.error("Please upload a CSV file first.")
        st.stop()
    
    try:
        df = pd.read_csv('data.csv')
        
        # Clean column names (strip whitespace and special characters)
        df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)
        
        # Check for required columns (after cleaning)
        required_columns = ['Principal Amount', 'Expected Repayment', 'Total Outstanding', 'Total Repayment', 
                          'Branch Name', 'Loan Officer Name', 'Client Name', 'Disbursed On Date']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            st.error("Please ensure your CSV file contains all required columns.")
            st.error(f"Available columns: {', '.join(df.columns)}")
            st.stop()
        
        # Remove rows that are repeated headers or have missing Principal Amount
        df = df[df['Principal Amount'].apply(lambda x: str(x).replace('.', '', 1).isdigit())]
        
        # Strip whitespace from string columns
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype(str).str.strip()
        
        # Parse dates
        df['Disbursed On Date'] = pd.to_datetime(df['Disbursed On Date'], errors='coerce')
        
        # Convert numeric columns
        for col in ['Principal Amount', 'Expected Repayment', 'Total Outstanding', 'Total Repayment']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with NaN in key numeric columns
        df = df.dropna(subset=['Principal Amount', 'Expected Repayment', 'Total Outstanding', 'Total Repayment'])
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        if df.empty:
            st.error("No valid data found after cleaning. Please check your CSV file.")
            st.stop()
            
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.error("Please ensure your CSV file is properly formatted.")
        st.stop()

# Only proceed with analysis if file is uploaded
if uploaded_file is not None or os.path.exists('zidisha loans July.csv'):
    # Load data only if not already loaded or if new file uploaded
    if not st.session_state.data_loaded or st.session_state.df is None:
        with st.spinner("Loading and processing data..."):
            st.session_state.df = load_data()
            st.session_state.data_loaded = True
    
    df = st.session_state.df

    # Sidebar filters
    branches = ['All'] + sorted(df['Branch Name'].dropna().unique())
    officers = ['All'] + sorted(df['Loan Officer Name'].dropna().unique())

    selected_branch = st.sidebar.selectbox('Select Branch', branches)
    selected_officer = st.sidebar.selectbox('Select Loan Officer', officers)

    filtered_df = df.copy()
    if selected_branch != 'All':
        filtered_df = filtered_df[filtered_df['Branch Name'] == selected_branch]
    if selected_officer != 'All':
        filtered_df = filtered_df[filtered_df['Loan Officer Name'] == selected_officer]

    # Metrics for cards
    total_disbursed = filtered_df['Principal Amount'].sum()
    total_repayment = filtered_df['Total Repayment'].sum()
    total_outstanding = filtered_df['Total Outstanding'].sum()
    num_loans = filtered_df.shape[0]
    avg_loan_size = filtered_df['Principal Amount'].mean()
    repayment_rate = (total_repayment / total_disbursed * 100) if total_disbursed > 0 else 0
    total_clients = filtered_df['Client Name'].nunique()

    # Calculate X and loans below expected repayment
    now = pd.Timestamp(datetime.now())
    filtered_df['Days Since Disbursement'] = (now - filtered_df['Disbursed On Date']).dt.days.clip(lower=0)
    filtered_df['X Fraction'] = filtered_df['Days Since Disbursement'] / 30
    filtered_df['Expected Paid By Now'] = filtered_df['X Fraction'] * filtered_df['Expected Repayment']
    loans_below_expected = filtered_df[filtered_df['Total Repayment'] < filtered_df['Expected Paid By Now']].shape[0]

    # Calculate number of loans not paid at all
    loans_not_paid = filtered_df[filtered_df['Total Repayment'] == 0].shape[0]

    # Calculate expected repayment as of today
    expected_repayment_today = int(filtered_df['Expected Paid By Now'].sum()) if 'Expected Paid By Now' in filtered_df.columns else 0

    # First row of cards
    row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
    row1_col1.metric('Total Disbursed', f"{int(total_disbursed)}")
    row1_col2.metric('Total Repayment', f"{int(total_repayment)}")
    row1_col3.metric('Total Outstanding', f"{int(total_outstanding)}")
    row1_col4.metric('Expected Repayment as of Today', f"{expected_repayment_today}")

    # Second row of cards
    row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
    row2_col1.metric('Number of Loans', num_loans)
    row2_col2.metric('Average Loan Size', f"{int(avg_loan_size) if not pd.isna(avg_loan_size) else 0}")
    row2_col3.metric('Loans Below Expected Repayment', loans_below_expected)
    row2_col4.metric('Loans Not Paid At All', loans_not_paid)

    # Plot: Disbursement vs Repayment
    st.subheader('Disbursement vs Repayment')

    # Prepare data for bar plot
    bar_df = filtered_df[['Branch Name', 'Principal Amount', 'Total Repayment']].copy()
    bar_df = bar_df.groupby('Branch Name', as_index=False).sum()
    bar_df = bar_df.sort_values('Principal Amount', ascending=False)

    bar_df_melted = bar_df.melt(id_vars='Branch Name', value_vars=['Principal Amount', 'Total Repayment'],
                                var_name='Type', value_name='Amount')

    fig = px.bar(
        bar_df_melted,
        x='Branch Name',
        y='Amount',
        color='Type',
        barmode='group',
        labels={'Amount': 'Ksh Amount'},
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    # Additional Visualizations
    st.subheader('📊 Additional Analytics')

    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Loan Officer Performance", "Repayment Analysis", "Loan Size Distribution", "Disbursement Trends", "Branch Comparison"])

    with tab1:
        st.write("**Loan Officer Performance Analysis**")
        
        # Loan Officer Performance - Total Disbursed vs Repayment Rate
        officer_performance = filtered_df.groupby('Loan Officer Name').agg({
            'Principal Amount': 'sum',
            'Total Repayment': 'sum',
            'Client Name': 'count'
        }).reset_index()
        
        officer_performance['Repayment Rate (%)'] = (officer_performance['Total Repayment'] / officer_performance['Principal Amount'] * 100).round(2)
        officer_performance['Average Loan Size'] = (officer_performance['Principal Amount'] / officer_performance['Client Name']).round(0)
        
        # Sort by repayment rate
        officer_performance = officer_performance.sort_values('Repayment Rate (%)', ascending=False)
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Top Performer", officer_performance.iloc[0]['Loan Officer Name'])
        with col2:
            st.metric("Best Repayment Rate", f"{officer_performance.iloc[0]['Repayment Rate (%)']:.1f}%")
        with col3:
            st.metric("Total Officers", len(officer_performance))
        
        # Performance chart
        fig_officer = px.bar(officer_performance.head(10), 
                           x='Loan Officer Name', 
                           y='Repayment Rate (%)',
                           title='Top 10 Loan Officers by Repayment Rate',
                           color='Principal Amount',
                           color_continuous_scale='viridis')
        fig_officer.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_officer, use_container_width=True)

    with tab2:
        st.write("**Repayment Analysis**")
        
        # Repayment Rate Analysis
        filtered_df['Repayment Rate'] = (filtered_df['Total Repayment'] / filtered_df['Principal Amount'] * 100).round(2)
        
        # Repayment rate distribution
        fig_repayment_dist = px.histogram(filtered_df, 
                                        x='Repayment Rate', 
                                        nbins=20,
                                        title='Distribution of Repayment Rates',
                                        labels={'Repayment Rate': 'Repayment Rate (%)'})
        st.plotly_chart(fig_repayment_dist, use_container_width=True)
        
        # Outstanding vs Repayment scatter
        fig_scatter = px.scatter(filtered_df, 
                               x='Principal Amount', 
                               y='Total Outstanding',
                               color='Repayment Rate',
                               size='Total Repayment',
                               hover_data=['Client Name', 'Loan Officer Name'],
                               title='Loan Amount vs Outstanding Amount (colored by Repayment Rate)')
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab3:
        st.write("**Loan Size Analysis**")
        
        # Loan size distribution
        fig_loan_dist = px.histogram(filtered_df, 
                                   x='Principal Amount', 
                                   nbins=20,
                                   title='Distribution of Loan Sizes',
                                   labels={'Principal Amount': 'Loan Amount (KSH)'})
        st.plotly_chart(fig_loan_dist, use_container_width=True)
        
        # Loan size categories
        filtered_df['Loan Size Category'] = pd.cut(filtered_df['Principal Amount'], 
                                                 bins=[0, 50000, 100000, 200000, float('inf')],
                                                 labels=['Small (<50K)', 'Medium (50K-100K)', 'Large (100K-200K)', 'Very Large (>200K)'])
        
        size_analysis = filtered_df.groupby('Loan Size Category').agg({
            'Principal Amount': 'sum',
            'Total Repayment': 'sum',
            'Client Name': 'count'
        }).reset_index()
        
        size_analysis['Repayment Rate (%)'] = (size_analysis['Total Repayment'] / size_analysis['Principal Amount'] * 100).round(2)
        
        fig_size = px.bar(size_analysis, 
                         x='Loan Size Category', 
                         y='Repayment Rate (%)',
                         title='Repayment Rate by Loan Size Category',
                         color='Principal Amount',
                         color_continuous_scale='plasma')
        st.plotly_chart(fig_size, use_container_width=True)

    with tab4:
        st.write("**Disbursement Trends**")
        
        # Debug date information
        st.write(f"**Debug Info:**")
        st.write(f"Total records: {len(filtered_df)}")
        st.write(f"Records with valid dates: {filtered_df['Disbursed On Date'].notna().sum()}")
        st.write(f"Date range: {filtered_df['Disbursed On Date'].min()} to {filtered_df['Disbursed On Date'].max()}")
        
        # Check if we have valid dates
        if filtered_df['Disbursed On Date'].notna().sum() > 0:
            # Daily disbursement trend (since all data is from July 2025)
            filtered_df['Date'] = filtered_df['Disbursed On Date'].dt.date
            daily_trend = filtered_df.groupby('Date').agg({
                'Principal Amount': 'sum',
                'Client Name': 'count'
            }).reset_index()
            daily_trend = daily_trend.sort_values('Date')
            
            if len(daily_trend) > 1:
                fig_daily_trend = px.line(daily_trend, 
                                        x='Date', 
                                        y='Principal Amount',
                                        title='Daily Disbursement Trend (July 2025)',
                                        labels={'Principal Amount': 'Total Disbursed (KSH)', 'Date': 'Date'})
                
                # Make the line smooth
                fig_daily_trend.update_traces(
                    line=dict(width=3, shape='spline'),
                    mode='lines+markers',
                    marker=dict(size=6)
                )
                
                # Improve layout
                fig_daily_trend.update_layout(
                    xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
                    plot_bgcolor='white',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_daily_trend, use_container_width=True)
            else:
                st.warning("Not enough daily data points for trend analysis.")
                
            # Also show monthly summary for July
            filtered_df['Month'] = filtered_df['Disbursed On Date'].dt.to_period('M')
            monthly_summary = filtered_df.groupby('Month').agg({
                'Principal Amount': 'sum',
                'Client Name': 'count'
            }).reset_index()
            
            if len(monthly_summary) > 0:
                st.write("**July 2025 Summary:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Disbursed in July", f"{int(monthly_summary.iloc[0]['Principal Amount'])} KSH")
                with col2:
                    st.metric("Total Loans in July", int(monthly_summary.iloc[0]['Client Name']))
                with col3:
                    st.metric("Average Daily Disbursement", f"{int(monthly_summary.iloc[0]['Principal Amount'] / len(daily_trend))} KSH")
        else:
            st.error("No valid dates found in 'Disbursed On Date' column. Please check your data format.")
        
        # Daily disbursement pattern (only if we have valid dates)
        if filtered_df['Disbursed On Date'].notna().sum() > 0:
            filtered_df['DayOfWeek'] = filtered_df['Disbursed On Date'].dt.day_name()
            daily_pattern = filtered_df.groupby('DayOfWeek').agg({
                'Principal Amount': 'sum',
                'Client Name': 'count'
            }).reset_index()
            
            if len(daily_pattern) > 0:
                # Reorder days
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                daily_pattern['DayOfWeek'] = pd.Categorical(daily_pattern['DayOfWeek'], categories=day_order, ordered=True)
                daily_pattern = daily_pattern.sort_values('DayOfWeek')
                
                fig_daily = px.bar(daily_pattern, 
                                  x='DayOfWeek', 
                                  y='Principal Amount',
                                  title='Disbursement by Day of Week',
                                  labels={'Principal Amount': 'Total Disbursed (KSH)', 'DayOfWeek': 'Day of Week'})
                st.plotly_chart(fig_daily, use_container_width=True)
            else:
                st.warning("No daily pattern data available.")
        else:
            st.error("Cannot create daily pattern without valid dates.")

    with tab5:
        st.write("**Branch Performance Comparison**")
        
        # Branch performance metrics
        branch_performance = filtered_df.groupby('Branch Name').agg({
            'Principal Amount': 'sum',
            'Total Repayment': 'sum',
            'Total Outstanding': 'sum',
            'Client Name': 'count'
        }).reset_index()
        
        branch_performance['Repayment Rate (%)'] = (branch_performance['Total Repayment'] / branch_performance['Principal Amount'] * 100).round(2)
        branch_performance['Average Loan Size'] = (branch_performance['Principal Amount'] / branch_performance['Client Name']).round(0)
        
        # Branch comparison chart
        fig_branch_comp = px.scatter(branch_performance, 
                                   x='Principal Amount', 
                                   y='Repayment Rate (%)',
                                   size='Client Name',
                                   color='Total Outstanding',
                                   hover_data=['Branch Name'],
                                   title='Branch Performance: Total Disbursed vs Repayment Rate',
                                   labels={'Principal Amount': 'Total Disbursed (KSH)', 'Repayment Rate (%)': 'Repayment Rate (%)'})
        st.plotly_chart(fig_branch_comp, use_container_width=True)
        
        # Branch ranking table
        st.write("**Branch Performance Ranking**")
        branch_ranking = branch_performance[['Branch Name', 'Principal Amount', 'Repayment Rate (%)', 'Client Name', 'Average Loan Size']].sort_values('Repayment Rate (%)', ascending=False)
        st.dataframe(branch_ranking, use_container_width=True)
else:
    st.info("Please upload a CSV file to begin the analysis.")