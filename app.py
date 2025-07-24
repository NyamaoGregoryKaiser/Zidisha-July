import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# Load data
def load_data():
    df = pd.read_csv('zidisha loans July.csv')
    # Clean column names (strip whitespace and special characters)
    df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)
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
    return df

df = load_data()

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

# Dashboard title
st.title('Zidisha Loans July Analysis')

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

# Trend Visualization for the Past 1 Week
st.subheader('Disbursement and Repayment Trend (Last 7 Days)')

# Get current date (2025-07-16 from user system)
current_date = pd.Timestamp('2025-07-16')
last_7_days = [current_date - pd.Timedelta(days=i) for i in range(6, -1, -1)]

# Prepare daily disbursement
trend_df = filtered_df.copy()
trend_df['Disbursed On Date'] = pd.to_datetime(trend_df['Disbursed On Date'], errors='coerce')

# Group by disbursement date for the last 7 days
mask = trend_df['Disbursed On Date'].isin(last_7_days)
daily_disbursed = trend_df[mask].groupby('Disbursed On Date')['Principal Amount'].sum().reindex(last_7_days, fill_value=0)

# For repayment, since there is no repayment date, we will show cumulative repayment for loans disbursed on each day
# (If you have a repayment date column, replace this logic accordingly)
daily_repayment = trend_df[mask].groupby('Disbursed On Date')['Total Repayment'].sum().reindex(last_7_days, fill_value=0)

trend_plot_df = pd.DataFrame({
    'Date': [d.date() for d in last_7_days],
    'Disbursed': daily_disbursed.values,
    'Repaid': daily_repayment.values
})

fig_trend = px.line(trend_plot_df, x='Date', y=['Disbursed', 'Repaid'], markers=True,
                    labels={'value': 'Ksh Amount', 'variable': 'Type'},
                    title='Disbursement and Repayment Trend (Last 7 Days)')
fig_trend.update_layout(xaxis_title='Date', yaxis_title='Ksh Amount')
st.plotly_chart(fig_trend, use_container_width=True) 