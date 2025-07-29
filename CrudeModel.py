import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class CrudePipelineAnalyzer:
    def __init__(self, csv_file):
        """
        Initialize the analyzer with the pipeline data CSV file
        """
        self.csv_file = csv_file
        self.data = None
        self.processed_data = None
        self.models = {}
        
    def load_and_explore_data(self):
        """
        Load the CSV file and perform initial data exploration
        """
        print("🔍 Loading and exploring pipeline data...")
        
        # Load the CSV with proper encoding handling
        try:
            self.data = pd.read_csv(self.csv_file, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                self.data = pd.read_csv(self.csv_file, encoding='latin-1')
            except UnicodeDecodeError:
                self.data = pd.read_csv(self.csv_file, encoding='cp1252')
        
        print(f"📊 Dataset Shape: {self.data.shape}")
        print(f"📈 Columns: {list(self.data.columns)}")
        print("\n" + "="*60)
        print("PIPELINE DATA OVERVIEW")
        print("="*60)
        
        # Get unique companies
        unique_companies = self.data['Pipe Name'].nunique()
        print(f"🏭 Total Pipeline Companies: {unique_companies}")
        
        # Get data types for each metric
        unique_items = self.data['Item'].unique()
        print(f"\n📋 Data Types Available ({len(unique_items)} types):")
        for i, item in enumerate(unique_items, 1):
            print(f"{i:2d}. {item}")
        
        # Check timeframe
        year_columns = [col for col in self.data.columns if col.endswith('Q4') or col.isdigit()]
        print(f"\n📅 Time Period: {year_columns[0]} to {year_columns[-1]} ({len(year_columns)} quarters)")
        
        return self.data
    
    def analyze_key_metrics(self):
        """
        Analyze key financial and operational metrics
        """
        print("\n" + "="*60)
        print("KEY METRICS ANALYSIS")
        print("="*60)
        
        # Key metrics to focus on
        key_metrics = {
            'Operating and Maintenance Expenses': 10,
            'Total Interstate Operating Revenues': 100,
            'Total Interstate Throughput in Barrels': 110,
            'Total Interstate Throughput in Barrel-Miles': 120,
            'Total Cost of Service': 90,
            'Rate of Return %': 60
        }
        
        analysis_results = {}
        
        for metric_name, row_num in key_metrics.items():
            metric_data = self.data[self.data['Row Number'] == row_num].copy()
            
            if not metric_data.empty:
                print(f"\n📊 {metric_name}:")
                
                # Get numeric columns (year data)
                year_cols = [col for col in metric_data.columns if 'Q4' in col or col.isdigit()]
                
                # Calculate statistics across all companies and years
                all_values = []
                for col in year_cols:
                    values = pd.to_numeric(metric_data[col], errors='coerce').dropna()
                    all_values.extend(values.tolist())
                
                if all_values:
                    all_values = np.array(all_values)
                    print(f"   📈 Mean: {np.mean(all_values):,.0f}")
                    print(f"   📊 Median: {np.median(all_values):,.0f}")
                    print(f"   📏 Std Dev: {np.std(all_values):,.0f}")
                    print(f"   ⬆️  Max: {np.max(all_values):,.0f}")
                    print(f"   ⬇️  Min: {np.min(all_values):,.0f}")
                    
                    analysis_results[metric_name] = {
                        'data': metric_data,
                        'stats': {
                            'mean': np.mean(all_values),
                            'median': np.median(all_values),
                            'std': np.std(all_values),
                            'max': np.max(all_values),
                            'min': np.min(all_values)
                        }
                    }
        
        return analysis_results
    
    def identify_top_companies(self):
        """
        Identify top performing companies by various metrics
        """
        print("\n" + "="*60)
        print("TOP COMPANIES ANALYSIS")
        print("="*60)
        
        # Get revenue data
        revenue_data = self.data[self.data['Row Number'] == 100].copy()
        throughput_data = self.data[self.data['Row Number'] == 110].copy()
        
        if not revenue_data.empty:
            print("\n🏆 TOP COMPANIES BY REVENUE (2024Q4):")
            
            # Focus on most recent year with data
            recent_col = '2024Q4'
            if recent_col in revenue_data.columns:
                revenue_2024 = revenue_data[['Pipe Name', 'Pipe Ownership', recent_col]].copy()
                revenue_2024[recent_col] = pd.to_numeric(revenue_2024[recent_col], errors='coerce')
                revenue_2024 = revenue_2024.dropna().sort_values(recent_col, ascending=False)
                
                for i, (_, row) in enumerate(revenue_2024.head(10).iterrows(), 1):
                    print(f"{i:2d}. {row['Pipe Name'][:40]:<40} ${row[recent_col]:>12,.0f}")
                    print(f"    Ownership: {row['Pipe Ownership']}")
        
        if not throughput_data.empty:
            print(f"\n🛢️ TOP COMPANIES BY THROUGHPUT ({recent_col}):")
            
            if recent_col in throughput_data.columns:
                throughput_2024 = throughput_data[['Pipe Name', 'Pipe Ownership', recent_col]].copy()
                throughput_2024[recent_col] = pd.to_numeric(throughput_2024[recent_col], errors='coerce')
                throughput_2024 = throughput_2024.dropna().sort_values(recent_col, ascending=False)
                
                for i, (_, row) in enumerate(throughput_2024.head(10).iterrows(), 1):
                    print(f"{i:2d}. {row['Pipe Name'][:40]:<40} {row[recent_col]:>15,.0f} barrels")
    
    def prepare_predictive_features(self):
        """
        Prepare data for machine learning models
        """
        print("\n" + "="*60)
        print("PREPARING PREDICTIVE FEATURES")
        print("="*60)
        
        # Create a comprehensive dataset for modeling
        modeling_data = []
        
        # Key metrics we want to use for prediction
        feature_metrics = {
            'opex': 10,  # Operating expenses
            'revenue': 100,  # Revenue
            'throughput_barrels': 110,  # Throughput in barrels
            'throughput_miles': 120,  # Throughput in barrel-miles
            'cost_of_service': 90,  # Total cost of service
            'depreciation': 20,  # Depreciation expense
            'rate_of_return': 60  # Rate of return
        }
        
        # Get all year columns
        year_cols = [col for col in self.data.columns if 'Q4' in col]
        
        for _, company_data in self.data.groupby('Pipe Name'):
            company_name = company_data['Pipe Name'].iloc[0]
            ownership = company_data['Pipe Ownership'].iloc[0]
            
            # Extract features for each year
            for year_col in year_cols:
                year = year_col.replace('Q4', '')
                
                row_dict = {
                    'company': company_name,
                    'ownership': ownership,
                    'year': int(year) if year.isdigit() else 0,
                    'quarter': 4  # All data is Q4
                }
                
                # Extract each metric for this company-year
                for metric_name, row_num in feature_metrics.items():
                    metric_row = company_data[company_data['Row Number'] == row_num]
                    if not metric_row.empty and year_col in metric_row.columns:
                        value = pd.to_numeric(metric_row[year_col].iloc[0], errors='coerce')
                        row_dict[metric_name] = value
                    else:
                        row_dict[metric_name] = np.nan
                
                modeling_data.append(row_dict)
        
        self.processed_data = pd.DataFrame(modeling_data)
        
        # Clean the data
        print(f"📊 Created modeling dataset: {self.processed_data.shape}")
        print(f"🏭 Companies: {self.processed_data['company'].nunique()}")
        print(f"📅 Years: {sorted(self.processed_data['year'].unique())}")
        
        # Remove rows where revenue is null (our main target)
        initial_rows = len(self.processed_data)
        self.processed_data = self.processed_data.dropna(subset=['revenue'])
        print(f"🧹 Removed {initial_rows - len(self.processed_data)} rows with missing revenue data")
        
        return self.processed_data
    
    def build_predictive_models(self):
        """
        Build machine learning models to predict revenue and throughput
        """
        print("\n" + "="*60)
        print("BUILDING PREDICTIVE MODELS")
        print("="*60)
        
        if self.processed_data is None:
            self.prepare_predictive_features()
        
        # Prepare features and targets
        feature_cols = ['opex', 'throughput_barrels', 'throughput_miles', 'cost_of_service', 'depreciation', 'year']
        
        # Remove rows with any missing feature data
        modeling_df = self.processed_data[feature_cols + ['revenue']].dropna()
        
        if len(modeling_df) < 50:
            print("❌ Insufficient data for modeling")
            return None
        
        X = modeling_df[feature_cols]
        y = modeling_df['revenue']
        
        print(f"🤖 Training models on {len(modeling_df)} data points")
        print(f"📊 Features: {feature_cols}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n🔧 Training {name}...")
            
            if name == 'Linear Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mse': mse,
                'r2': r2,
                'mae': mae,
                'predictions': y_pred
            }
            
            print(f"   📊 R² Score: {r2:.3f}")
            print(f"   📏 MAE: ${mae:,.0f}")
            print(f"   📈 RMSE: ${np.sqrt(mse):,.0f}")
        
        self.models = results
        return results
    
    def identify_trading_opportunities(self):
        """
        Identify potential trading opportunities based on the analysis
        """
        print("\n" + "="*60)
        print("TRADING OPPORTUNITIES & INSIGHTS")
        print("="*60)
        
        if self.processed_data is None:
            self.prepare_predictive_features()
        
        # 1. Growth Analysis
        print("\n📈 GROWTH ANALYSIS:")
        growth_analysis = self.processed_data.groupby('company').agg({
            'revenue': ['first', 'last'],
            'throughput_barrels': ['first', 'last'],
            'year': ['min', 'max']
        }).round(0)
        
        # Calculate growth rates
        companies_with_growth = []
        for company in self.processed_data['company'].unique():
            company_data = self.processed_data[self.processed_data['company'] == company].sort_values('year')
            if len(company_data) >= 2:
                first_revenue = company_data['revenue'].iloc[0]
                last_revenue = company_data['revenue'].iloc[-1]
                first_year = company_data['year'].iloc[0]
                last_year = company_data['year'].iloc[-1]
                
                if pd.notna(first_revenue) and pd.notna(last_revenue) and first_revenue > 0:
                    years_span = last_year - first_year
                    if years_span > 0:
                        cagr = ((last_revenue / first_revenue) ** (1/years_span)) - 1
                        companies_with_growth.append({
                            'company': company,
                            'cagr': cagr,
                            'first_revenue': first_revenue,
                            'last_revenue': last_revenue,
                            'years': years_span
                        })
        
        growth_df = pd.DataFrame(companies_with_growth).sort_values('cagr', ascending=False)
        
        print("🚀 TOP GROWTH COMPANIES (Revenue CAGR):")
        for i, row in growth_df.head(5).iterrows():
            print(f"   {row['company'][:50]:<50} {row['cagr']*100:>6.1f}% CAGR")
        
        print("\n📉 DECLINING COMPANIES (Potential Short Opportunities):")
        for i, row in growth_df.tail(5).iterrows():
            print(f"   {row['company'][:50]:<50} {row['cagr']*100:>6.1f}% CAGR")
        
        # 2. Efficiency Analysis
        print("\n⚡ EFFICIENCY ANALYSIS (Revenue per Barrel):")
        recent_data = self.processed_data[self.processed_data['year'] >= 2022]
        efficiency_data = recent_data.groupby('company').agg({
            'revenue': 'mean',
            'throughput_barrels': 'mean'
        }).dropna()
        
        efficiency_data['revenue_per_barrel'] = efficiency_data['revenue'] / efficiency_data['throughput_barrels']
        efficiency_data = efficiency_data.sort_values('revenue_per_barrel', ascending=False)
        
        print("💰 HIGHEST REVENUE PER BARREL (Premium Operations):")
        for i, (company, row) in enumerate(efficiency_data.head(5).iterrows(), 1):
            print(f"{i}. {company[:50]:<50} ${row['revenue_per_barrel']:.2f}/barrel")
        
        # 3. Market Share Analysis
        print("\n🏭 MARKET SHARE ANALYSIS (by Throughput):")
        market_share = recent_data.groupby('company')['throughput_barrels'].mean().sort_values(ascending=False)
        total_throughput = market_share.sum()
        
        print("📊 TOP MARKET SHARE HOLDERS:")
        for i, (company, throughput) in enumerate(market_share.head(5).items(), 1):
            share = (throughput / total_throughput) * 100
            print(f"{i}. {company[:50]:<50} {share:>6.1f}% ({throughput:>15,.0f} barrels)")
        
        return {
            'growth_companies': growth_df,
            'efficiency_data': efficiency_data,
            'market_share': market_share
        }
    
    def generate_trading_recommendations(self):
        """
        Generate specific trading recommendations based on the analysis
        """
        print("\n" + "="*60)
        print("TRADING RECOMMENDATIONS")
        print("="*60)
        
        opportunities = self.identify_trading_opportunities()
        
        print("\n🎯 LONG POSITION OPPORTUNITIES:")
        print("   Based on growth, efficiency, and market position")
        
        # Get top growth companies
        top_growth = opportunities['growth_companies'].head(3)
        
        for i, (_, company) in enumerate(top_growth.iterrows(), 1):
            print(f"\n{i}. 📈 {company['company']}")
            print(f"   💹 Revenue CAGR: {company['cagr']*100:.1f}%")
            print(f"   💰 Revenue Growth: ${company['first_revenue']:,.0f} → ${company['last_revenue']:,.0f}")
            print(f"   📅 Time Period: {company['years']} years")
            print(f"   🎯 Strategy: Long position on parent company stocks")
        
        print("\n🎯 SHORT POSITION OPPORTUNITIES:")
        print("   Based on declining performance and efficiency")
        
        # Get declining companies
        declining = opportunities['growth_companies'].tail(3)
        
        for i, (_, company) in enumerate(declining.iterrows(), 1):
            print(f"\n{i}. 📉 {company['company']}")
            print(f"   📉 Revenue CAGR: {company['cagr']*100:.1f}%")
            print(f"   💸 Revenue Decline: ${company['first_revenue']:,.0f} → ${company['last_revenue']:,.0f}")
            print(f"   ⚠️  Strategy: Consider short positions or avoid")
        
        print("\n💡 SECTOR INSIGHTS:")
        print("   1. 🛢️  Focus on companies with consistent throughput growth")
        print("   2. 💰 High revenue-per-barrel companies show pricing power")
        print("   3. 📊 Market leaders have competitive moats")
        print("   4. ⚡ Efficiency improvements indicate operational excellence")
        print("   5. 🔄 Monitor quarterly throughput changes for demand signals")
    
    def create_visualizations(self):
        """
        Create visualizations for the analysis
        """
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        if self.processed_data is None:
            self.prepare_predictive_features()
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Revenue trends over time
        revenue_trends = self.processed_data.groupby('year')['revenue'].agg(['mean', 'median']).dropna()
        axes[0,0].plot(revenue_trends.index, revenue_trends['mean'], marker='o', label='Mean Revenue')
        axes[0,0].plot(revenue_trends.index, revenue_trends['median'], marker='s', label='Median Revenue')
        axes[0,0].set_title('Pipeline Industry Revenue Trends')
        axes[0,0].set_xlabel('Year')
        axes[0,0].set_ylabel('Revenue ($)')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # 2. Throughput trends
        throughput_trends = self.processed_data.groupby('year')['throughput_barrels'].agg(['mean', 'median']).dropna()
        axes[0,1].plot(throughput_trends.index, throughput_trends['mean'], marker='o', label='Mean Throughput', color='orange')
        axes[0,1].plot(throughput_trends.index, throughput_trends['median'], marker='s', label='Median Throughput', color='red')
        axes[0,1].set_title('Pipeline Throughput Trends')
        axes[0,1].set_xlabel('Year')
        axes[0,1].set_ylabel('Throughput (Barrels)')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # 3. Revenue vs Throughput correlation
        recent_data = self.processed_data[self.processed_data['year'] >= 2020].dropna(subset=['revenue', 'throughput_barrels'])
        axes[1,0].scatter(recent_data['throughput_barrels'], recent_data['revenue'], alpha=0.6)
        axes[1,0].set_title('Revenue vs Throughput (2020+)')
        axes[1,0].set_xlabel('Throughput (Barrels)')
        axes[1,0].set_ylabel('Revenue ($)')
        axes[1,0].grid(True)
        
        # 4. Top companies by recent revenue
        recent_companies = recent_data.groupby('company')['revenue'].mean().sort_values(ascending=False).head(10)
        axes[1,1].barh(range(len(recent_companies)), recent_companies.values)
        axes[1,1].set_yticks(range(len(recent_companies)))
        axes[1,1].set_yticklabels([name[:20] + '...' if len(name) > 20 else name for name in recent_companies.index])
        axes[1,1].set_title('Top 10 Companies by Revenue (Recent)')
        axes[1,1].set_xlabel('Average Revenue ($)')
        
        plt.tight_layout()
        plt.savefig('pipeline_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 Visualizations saved as 'pipeline_analysis.png'")
        
        return fig
    
    def run_complete_analysis(self):
        """
        Run the complete analysis pipeline
        """
        print("🚀 STARTING COMPREHENSIVE CRUDE PIPELINE ANALYSIS")
        print("="*60)
        
        # Load and explore data
        self.load_and_explore_data()
        
        # Analyze key metrics
        self.analyze_key_metrics()
        
        # Identify top companies
        self.identify_top_companies()
        
        # Prepare data for modeling
        self.prepare_predictive_features()
        
        # Build predictive models
        self.build_predictive_models()
        
        # Identify trading opportunities
        self.identify_trading_opportunities()
        
        # Generate recommendations
        self.generate_trading_recommendations()
        
        # Create visualizations
        self.create_visualizations()
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE!")
        print("="*60)
        print("Next steps:")
        print("1. Review the trading recommendations above")
        print("2. Check the generated visualizations (pipeline_analysis.png)")
        print("3. Consider the parent companies for actual trading")
        print("4. Monitor quarterly updates for trend changes")


# Usage
if __name__ == "__main__":
    # Initialize the analyzer
    analyzer = CrudePipelineAnalyzer('EastDaley_Form6_pg700_2025_06(Form 6 Page 700).csv')
    
    # Run complete analysis
    analyzer.run_complete_analysis()