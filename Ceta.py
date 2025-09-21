import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class CETAImpactAnalyzer:
    def __init__(self, country_sector):
        self.country_sector = country_sector
        self.colors = ['#0055A4', '#FF0000', '#FFCC00', '#009900', '#660099', 
                      '#FF6600', '#0066CC', '#CC0000', '#00CCCC', '#FF00FF']
        
        self.start_year = 2017  # Entrée en vigueur provisoire du CETA
        self.end_year = 2027
        
        # Configuration spécifique pour chaque pays/secteur
        self.config = self._get_country_sector_config()
        
    def _get_country_sector_config(self):
        """Retourne la configuration spécifique pour chaque pays/secteur"""
        configs = {
            "France": {
                "type": "pays_ue",
                "pib_base": 2700000,
                "exportations_canada_base": 8500,
                "importations_canada_base": 7200,
                "secteurs_cles": ["aerospatial", "vin", "fromage", "luxe", "automobile"]
            },
            "Allemagne": {
                "type": "pays_ue",
                "pib_base": 3800000,
                "exportations_canada_base": 12500,
                "importations_canada_base": 9800,
                "secteurs_cles": ["automobile", "machines", "chimie", "electronique", "equipements"]
            },
            "Italie": {
                "type": "pays_ue",
                "pib_base": 2000000,
                "exportations_canada_base": 6800,
                "importations_canada_base": 5200,
                "secteurs_cles": ["mode", "agroalimentaire", "machines", "mobilier", "automobile"]
            },
            "Espagne": {
                "type": "pays_ue",
                "pib_base": 1400000,
                "exportations_canada_base": 4200,
                "importations_canada_base": 3800,
                "secteurs_cles": ["agroalimentaire", "automobile", "vin", "tourisme", "services"]
            },
            "Pays-Bas": {
                "type": "pays_ue",
                "pib_base": 900000,
                "exportations_canada_base": 5800,
                "importations_canada_base": 6200,
                "secteurs_cles": ["logistique", "agriculture", "energie", "chimie", "services_financiers"]
            },
            "Canada": {
                "type": "pays_partenaire",
                "pib_base": 1800000,
                "exportations_ue_base": 42000,
                "importations_ue_base": 45000,
                "secteurs_cles": ["agriculture", "energie", "automobile", "bois", "minerais"]
            },
            "UE-27": {
                "type": "union",
                "pib_base": 15500000,
                "exportations_canada_base": 420000,
                "importations_canada_base": 380000,
                "secteurs_cles": ["automobile", "agroalimentaire", "chimie", "machines", "services"]
            },
            "Agriculture": {
                "type": "secteur",
                "exportations_base": 8500,
                "importations_base": 7200,
                "pays_cles": ["France", "Allemagne", "Pays-Bas", "Italie", "Espagne"]
            },
            "Automobile": {
                "type": "secteur",
                "exportations_base": 32000,
                "importations_base": 28000,
                "pays_cles": ["Allemagne", "France", "Italie", "Espagne", "République tchèque"]
            },
            "Services": {
                "type": "secteur",
                "exportations_base": 28000,
                "importations_base": 25000,
                "pays_cles": ["France", "Allemagne", "Royaume-Uni", "Pays-Bas", "Irlande"]
            },
            # Configuration par défaut
            "default": {
                "type": "pays_ue",
                "pib_base": 500000,
                "exportations_canada_base": 2500,
                "importations_canada_base": 2200,
                "secteurs_cles": ["diversifies"]
            }
        }
        
        return configs.get(self.country_sector, configs["default"])
    
    def generate_ceta_data(self):
        """Génère des données sur l'impact du CETA"""
        print(f"🇪🇺🇨🇦 Génération des données CETA pour {self.country_sector}...")
        
        # Créer une base de données annuelle
        dates = pd.date_range(start=f'{self.start_year}-01-01', 
                             end=f'{self.end_year}-12-31', freq='Y')
        
        data = {'Annee': [date.year for date in dates]}
        
        # Données économiques de base
        data['PIB'] = self._simulate_gdp(dates)
        
        # Échanges commerciaux
        data['Exportations_Vers_Canada'] = self._simulate_exports(dates)
        data['Importations_Du_Canada'] = self._simulate_imports(dates)
        data['Balance_Commerciale'] = self._calculate_trade_balance(dates)
        
        # Droits de douane et barrières
        data['Droits_Douane_Moyens'] = self._simulate_tariffs(dates)
        data['Barrieres_Non_Tarifaires'] = self._simulate_non_tariff_barriers(dates)
        
        # Impacts sectoriels
        data['Creation_Emplois'] = self._simulate_job_creation(dates)
        data['Croissance_Sectorielle'] = self._simulate_sector_growth(dates)
        data['Investissements_Etrangers'] = self._simulate_foreign_investment(dates)
        
        # Indicateurs d'impact économique
        data['Impact_Sur_PIB'] = self._simulate_gdp_impact(dates)
        data['Gains_Productivite'] = self._simulate_productivity_gains(dates)
        data['Economies_Douanieres'] = self._simulate_customs_savings(dates)
        
        # Indicateurs spécifiques selon le type
        if self.config["type"] == "pays_ue":
            for secteur in self.config["secteurs_cles"]:
                if secteur == "vin":
                    data['Exportations_Vin'] = self._simulate_wine_exports(dates)
                elif secteur == "fromage":
                    data['Exportations_Fromage'] = self._simulate_cheese_exports(dates)
                elif secteur == "automobile":
                    data['Exportations_Automobile'] = self._simulate_auto_exports(dates)
                elif secteur == "aerospatial":
                    data['Exportations_Aerospatial'] = self._simulate_aerospace_exports(dates)
        
        elif self.config["type"] == "secteur":
            for pays in self.config["pays_cles"]:
                data[f'Exportations_{pays}'] = self._simulate_country_exports(dates, pays)
        
        df = pd.DataFrame(data)
        
        # Ajouter des tendances spécifiques
        self._add_ceta_trends(df)
        
        return df
    
    def _simulate_gdp(self, dates):
        """Simule l'évolution du PIB"""
        base_gdp = self.config["pib_base"]
        
        gdp = []
        for i, date in enumerate(dates):
            year = date.year
            
            # Croissance de base différente selon le type
            if self.config["type"] == "pays_ue":
                base_growth = 0.018  # Croissance moyenne UE
            elif self.config["type"] == "pays_partenaire":
                base_growth = 0.022  # Croissance moyenne Canada
            elif self.config["type"] == "union":
                base_growth = 0.019  # Croissance moyenne UE
            else:  # secteur
                base_growth = 0.020  # Croissance moyenne sectorielle
                
            # Effet du CETA sur la croissance
            ceta_effect = 0.0
            if year >= 2017:  # Après l'entrée en vigueur
                ceta_effect = 0.002 * (year - 2016)  # Effet cumulatif
                
            growth = 1 + (base_growth + ceta_effect) * i
            gdp.append(base_gdp * growth)
        
        return gdp
    
    def _simulate_exports(self, dates):
        """Simule les exportations vers le Canada"""
        if self.config["type"] == "pays_ue":
            base_exports = self.config["exportations_canada_base"]
        elif self.config["type"] == "secteur":
            base_exports = self.config["exportations_base"]
        else:  # Canada ou UE
            base_exports = self.config["exportations_ue_base"]
        
        exports = []
        for i, date in enumerate(dates):
            year = date.year
            
            # Effet du CETA sur les exportations
            ceta_effect = 0.0
            if year >= 2017:  # Après l'entrée en vigueur
                # Augmentation progressive des exportations grâce au CETA
                ceta_effect = 0.08 * min(5, year - 2016)  # Effet croissant jusqu'à 5 ans
                
            growth = 1 + (0.03 + ceta_effect) * i  # Croissance de base + effet CETA
            exports.append(base_exports * growth)
        
        return exports
    
    def _simulate_imports(self, dates):
        """Simule les importations depuis le Canada"""
        if self.config["type"] == "pays_ue":
            base_imports = self.config["importations_canada_base"]
        elif self.config["type"] == "secteur":
            base_imports = self.config["importations_base"]
        else:  # Canada ou UE
            base_imports = self.config["importations_ue_base"]
        
        imports = []
        for i, date in enumerate(dates):
            year = date.year
            
            # Effet du CETA sur les importations
            ceta_effect = 0.0
            if year >= 2017:  # Après l'entrée en vigueur
                # Augmentation progressive des importations grâce au CETA
                ceta_effect = 0.06 * min(5, year - 2016)  # Effet croissant jusqu'à 5 ans
                
            growth = 1 + (0.025 + ceta_effect) * i  # Croissance de base + effet CETA
            imports.append(base_imports * growth)
        
        return imports
    
    def _calculate_trade_balance(self, dates):
        """Calcule la balance commerciale"""
        exports = self._simulate_exports(dates)
        imports = self._simulate_imports(dates)
        
        return [exports[i] - imports[i] for i in range(len(exports))]
    
    def _simulate_tariffs(self, dates):
        """Simule l'évolution des droits de douane moyens"""
        tariffs = []
        for i, date in enumerate(dates):
            year = date.year
            
            if year < 2017:
                tariff = 4.2  # Moyenne avant CETA
            elif year < 2020:
                tariff = 1.8  # Réduction rapide
            elif year < 2023:
                tariff = 0.7  # Réduction continue
            else:
                tariff = 0.1  # Presque éliminés
                
            tariffs.append(tariff)
        
        return tariffs
    
    def _simulate_non_tariff_barriers(self, dates):
        """Simule la réduction des barrières non tarifaires"""
        barriers = []
        for i, date in enumerate(dates):
            year = date.year
            
            if year < 2017:
                barrier = 8.5  # Niveau élevé avant CETA
            elif year < 2020:
                barrier = 6.2  # Réduction modérée
            elif year < 2023:
                barrier = 4.1  # Réduction continue
            else:
                barrier = 2.8  # Niveau réduit mais persistant
                
            barriers.append(barrier)
        
        return barriers
    
    def _simulate_job_creation(self, dates):
        """Simule la création d'emplois liée au CETA"""
        jobs = []
        for i, date in enumerate(dates):
            year = date.year
            
            if year < 2017:
                created = 0  # Avant CETA
            elif year < 2020:
                created = 8000 * (year - 2016)  # Création progressive
            elif year < 2023:
                created = 24000 + 5000 * (year - 2019)  # Accélération
            else:
                created = 39000 + 3000 * (year - 2022)  # Stabilisation
                
            # Ajustement selon le type
            if self.config["type"] == "pays_ue":
                multiplier = 1.0
            elif self.config["type"] == "secteur":
                multiplier = 0.3
            else:
                multiplier = 0.1
                
            jobs.append(created * multiplier)
        
        return jobs
    
    def _simulate_sector_growth(self, dates):
        """Simule la croissance sectorielle liée au CETA"""
        growth = []
        for i, date in enumerate(dates):
            year = date.year
            
            if year < 2017:
                sector_growth = 0.0  # Avant CETA
            elif year < 2020:
                sector_growth = 0.015 * (year - 2016)  # Croissance progressive
            elif year < 2023:
                sector_growth = 0.045 + 0.008 * (year - 2019)  # Accélération
            else:
                sector_growth = 0.069 + 0.005 * (year - 2022)  # Stabilisation
                
            growth.append(sector_growth)
        
        return growth
    
    def _simulate_foreign_investment(self, dates):
        """Simule l'augmentation des investissements étrangers"""
        investment = []
        base_investment = 1000  # Base en millions d'euros
        
        for i, date in enumerate(dates):
            year = date.year
            
            if year < 2017:
                increase = 0  # Avant CETA
            elif year < 2020:
                increase = 0.12 * (year - 2016)  # Augmentation progressive
            elif year < 2023:
                increase = 0.36 + 0.10 * (year - 2019)  # Accélération
            else:
                increase = 0.66 + 0.08 * (year - 2022)  # Stabilisation
                
            investment.append(base_investment * (1 + increase))
        
        return investment
    
    def _simulate_gdp_impact(self, dates):
        """Simule l'impact du CETA sur le PIB"""
        impact = []
        for i, date in enumerate(dates):
            year = date.year
            
            if year < 2017:
                gdp_impact = 0.0  # Avant CETA
            elif year < 2020:
                gdp_impact = 0.0012 * (year - 2016)  # Impact progressif
            elif year < 2023:
                gdp_impact = 0.0036 + 0.0008 * (year - 2019)  # Accélération
            else:
                gdp_impact = 0.0060 + 0.0006 * (year - 2022)  # Stabilisation
                
            impact.append(gdp_impact)
        
        return impact
    
    def _simulate_productivity_gains(self, dates):
        """Simule les gains de productivité liés au CETA"""
        gains = []
        for i, date in enumerate(dates):
            year = date.year
            
            if year < 2017:
                productivity_gain = 0.0  # Avant CETA
            elif year < 2020:
                productivity_gain = 0.0008 * (year - 2016)  # Gains progressifs
            elif year < 2023:
                productivity_gain = 0.0024 + 0.0006 * (year - 2019)  # Accélération
            else:
                productivity_gain = 0.0042 + 0.0004 * (year - 2022)  # Stabilisation
                
            gains.append(productivity_gain)
        
        return gains
    
    def _simulate_customs_savings(self, dates):
        """Simule les économies douanières liées au CETA"""
        savings = []
        base_savings = 500  # Base en millions d'euros
        
        for i, date in enumerate(dates):
            year = date.year
            
            if year < 2017:
                saving = 0  # Avant CETA
            elif year < 2020:
                saving = 0.25 * (year - 2016)  # Économies progressives
            elif year < 2023:
                saving = 0.75 + 0.20 * (year - 2019)  # Accélération
            else:
                saving = 1.35 + 0.15 * (year - 2022)  # Stabilisation
                
            savings.append(base_savings * saving)
        
        return savings
    
    def _simulate_wine_exports(self, dates):
        """Simule les exportations de vin vers le Canada"""
        exports = []
        base_exports = 1200  # Millions d'euros
        
        for i, date in enumerate(dates):
            year = date.year
            
            if year < 2017:
                growth = 1.0  # Avant CETA
            elif year < 2020:
                growth = 1.0 + 0.15 * (year - 2016)  # Croissance progressive
            elif year < 2023:
                growth = 1.45 + 0.12 * (year - 2019)  # Accélération
            else:
                growth = 1.81 + 0.10 * (year - 2022)  # Stabilisation
                
            exports.append(base_exports * growth)
        
        return exports
    
    def _simulate_cheese_exports(self, dates):
        """Simule les exportations de fromage vers le Canada"""
        exports = []
        base_exports = 850  # Millions d'euros
        
        for i, date in enumerate(dates):
            year = date.year
            
            if year < 2017:
                growth = 1.0  # Avant CETA
            elif year < 2020:
                growth = 1.0 + 0.18 * (year - 2016)  # Croissance progressive
            elif year < 2023:
                growth = 1.54 + 0.15 * (year - 2019)  # Accélération
            else:
                growth = 1.99 + 0.12 * (year - 2022)  # Stabilisation
                
            exports.append(base_exports * growth)
        
        return exports
    
    def _simulate_auto_exports(self, dates):
        """Simule les exportations automobiles vers le Canada"""
        exports = []
        base_exports = 5800  # Millions d'euros
        
        for i, date in enumerate(dates):
            year = date.year
            
            if year < 2017:
                growth = 1.0  # Avant CETA
            elif year < 2020:
                growth = 1.0 + 0.10 * (year - 2016)  # Croissance progressive
            elif year < 2023:
                growth = 1.30 + 0.08 * (year - 2019)  # Accélération
            else:
                growth = 1.54 + 0.06 * (year - 2022)  # Stabilisation
                
            exports.append(base_exports * growth)
        
        return exports
    
    def _simulate_aerospace_exports(self, dates):
        """Simule les exportations aérospatiales vers le Canada"""
        exports = []
        base_exports = 4200  # Millions d'euros
        
        for i, date in enumerate(dates):
            year = date.year
            
            if year < 2017:
                growth = 1.0  # Avant CETA
            elif year < 2020:
                growth = 1.0 + 0.12 * (year - 2016)  # Croissance progressive
            elif year < 2023:
                growth = 1.36 + 0.10 * (year - 2019)  # Accélération
            else:
                growth = 1.66 + 0.08 * (year - 2022)  # Stabilisation
                
            exports.append(base_exports * growth)
        
        return exports
    
    def _simulate_country_exports(self, dates, country):
        """Simule les exportations d'un pays spécifique pour un secteur"""
        exports = []
        base_exports = 1500  # Millions d'euros
        
        for i, date in enumerate(dates):
            year = date.year
            
            if year < 2017:
                growth = 1.0  # Avant CETA
            elif year < 2020:
                growth = 1.0 + 0.14 * (year - 2016)  # Croissance progressive
            elif year < 2023:
                growth = 1.42 + 0.11 * (year - 2019)  # Accélération
            else:
                growth = 1.75 + 0.09 * (year - 2022)  # Stabilisation
                
            exports.append(base_exports * growth)
        
        return exports
    
    def _add_ceta_trends(self, df):
        """Ajoute des tendances spécifiques liées au CETA"""
        for i, row in df.iterrows():
            year = row['Annee']
            
            # Effets de l'entrée en vigueur provisoire (2017)
            if year >= 2017:
                df.loc[i, 'Exportations_Vers_Canada'] *= 1.08  # Augmentation initiale
                df.loc[i, 'Importations_Du_Canada'] *= 1.06    # Augmentation initiale
            
            # Effets de la ratification complète (hypothétique 2020)
            if year >= 2020:
                df.loc[i, 'Exportations_Vers_Canada'] *= 1.12  # Augmentation supplémentaire
                df.loc[i, 'Importations_Du_Canada'] *= 1.09    # Augmentation supplémentaire
                df.loc[i, 'Investissements_Etrangers'] *= 1.15 # Augmentation des investissements
            
            # Impact de la pandémie COVID-19 (2020-2021)
            if 2020 <= year <= 2021:
                df.loc[i, 'Exportations_Vers_Canada'] *= 0.85  # Réduction due au COVID
                df.loc[i, 'Importations_Du_Canada'] *= 0.88    # Réduction due au COVID
            
            # Reprise post-COVID (2022-2023)
            if year >= 2022:
                df.loc[i, 'Exportations_Vers_Canada'] *= 1.18  # Reprise forte
                df.loc[i, 'Importations_Du_Canada'] *= 1.15    # Reprise forte
    
    def create_ceta_analysis(self, df):
        """Crée une analyse complète de l'impact du CETA"""
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Évolution des échanges commerciaux
        ax1 = plt.subplot(4, 2, 1)
        self._plot_trade_evolution(df, ax1)
        
        # 2. Impact sur la balance commerciale
        ax2 = plt.subplot(4, 2, 2)
        self._plot_trade_balance(df, ax2)
        
        # 3. Réduction des barrières commerciales
        ax3 = plt.subplot(4, 2, 3)
        self._plot_trade_barriers(df, ax3)
        
        # 4. Impact sur l'emploi et les investissements
        ax4 = plt.subplot(4, 2, 4)
        self._plot_employment_investment(df, ax4)
        
        # 5. Impact sur le PIB et la productivité
        ax5 = plt.subplot(4, 2, 5)
        self._plot_gdp_productivity(df, ax5)
        
        # 6. Analyse sectorielle
        ax6 = plt.subplot(4, 2, 6)
        self._plot_sectoral_analysis(df, ax6)
        
        # 7. Gains économiques
        ax7 = plt.subplot(4, 2, 7)
        self._plot_economic_gains(df, ax7)
        
        # 8. Comparaison avant/après CETA
        ax8 = plt.subplot(4, 2, 8)
        self._plot_before_after_comparison(df, ax8)
        
        plt.suptitle(f'Analyse de l\'Impact du CETA - {self.country_sector} ({self.start_year}-{self.end_year})', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.country_sector}_ceta_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Générer les insights
        self._generate_ceta_insights(df)
    
    def _plot_trade_evolution(self, df, ax):
        """Plot de l'évolution des échanges commerciaux"""
        ax.plot(df['Annee'], df['Exportations_Vers_Canada'], label='Exportations vers Canada', 
               linewidth=2, color='#0055A4', alpha=0.8)
        ax.plot(df['Annee'], df['Importations_Du_Canada'], label='Importations depuis Canada', 
               linewidth=2, color='#FF0000', alpha=0.8)
        
        ax.set_title('Évolution des Échanges Commerciaux (M€)', 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Valeur (M€)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_trade_balance(self, df, ax):
        """Plot de l'impact sur la balance commerciale"""
        ax.bar(df['Annee'], df['Balance_Commerciale'], 
              color=['#009900' if x >= 0 else '#CC0000' for x in df['Balance_Commerciale']],
              alpha=0.7)
        
        ax.set_title('Balance Commerciale avec le Canada (M€)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Balance (M€)')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_trade_barriers(self, df, ax):
        """Plot de la réduction des barrières commerciales"""
        ax.plot(df['Annee'], df['Droits_Douane_Moyens'], label='Droits de douane moyens (%)', 
               linewidth=2, color='#0055A4', alpha=0.8)
        ax.plot(df['Annee'], df['Barrieres_Non_Tarifaires'], label='Barrières non tarifaires (indice)', 
               linewidth=2, color='#FF6600', alpha=0.8)
        
        ax.set_title('Réduction des Barrières Commerciales', fontsize=12, fontweight='bold')
        ax.set_ylabel('Niveau')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_employment_investment(self, df, ax):
        """Plot de l'impact sur l'emploi et les investissements"""
        ax.plot(df['Annee'], df['Creation_Emplois'], label='Création d\'emplois', 
               linewidth=2, color='#0055A4', alpha=0.8)
        
        ax.set_title('Impact sur l\'Emploi', fontsize=12, fontweight='bold')
        ax.set_ylabel('Emplois créés', color='#0055A4')
        ax.tick_params(axis='y', labelcolor='#0055A4')
        ax.grid(True, alpha=0.3)
        
        # Investissements en second axe
        ax2 = ax.twinx()
        ax2.plot(df['Annee'], df['Investissements_Etrangers'], label='Investissements étrangers (M€)', 
                linewidth=2, color='#FF0000', alpha=0.8)
        ax2.set_ylabel('Investissements (M€)', color='#FF0000')
        ax2.tick_params(axis='y', labelcolor='#FF0000')
        
        # Combiner les légendes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    def _plot_gdp_productivity(self, df, ax):
        """Plot de l'impact sur le PIB et la productivité"""
        ax.plot(df['Annee'], df['Impact_Sur_PIB'], label='Impact sur le PIB (%)', 
               linewidth=2, color='#0055A4', alpha=0.8)
        
        ax.set_title('Impact sur le PIB et la Productivité', fontsize=12, fontweight='bold')
        ax.set_ylabel('Impact sur PIB (%)', color='#0055A4')
        ax.tick_params(axis='y', labelcolor='#0055A4')
        ax.grid(True, alpha=0.3)
        
        # Gains de productivité en second axe
        ax2 = ax.twinx()
        ax2.plot(df['Annee'], df['Gains_Productivite'], label='Gains de productivité (%)', 
                linewidth=2, color='#009900', alpha=0.8)
        ax2.set_ylabel('Gains productivité (%)', color='#009900')
        ax2.tick_params(axis='y', labelcolor='#009900')
        
        # Combiner les légendes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    def _plot_sectoral_analysis(self, df, ax):
        """Plot de l'analyse sectorielle"""
        # Sélectionner les indicateurs sectoriels disponibles
        sector_columns = [col for col in df.columns if col.startswith('Exportations_') and col != 'Exportations_Vers_Canada']
        
        colors = ['#0055A4', '#FF0000', '#FFCC00', '#009900', '#660099']
        
        for i, column in enumerate(sector_columns[:5]):  # Limiter à 5 secteurs
            sector_name = column.replace('Exportations_', '')
            ax.plot(df['Annee'], df[column], label=sector_name, 
                   linewidth=2, color=colors[i % len(colors)], alpha=0.8)
        
        ax.set_title('Analyse Sectorielle - Exportations (M€)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Valeur (M€)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_economic_gains(self, df, ax):
        """Plot des gains économiques"""
        ax.plot(df['Annee'], df['Economies_Douanieres'], label='Économies douanières (M€)', 
               linewidth=2, color='#0055A4', alpha=0.8)
        
        ax.set_title('Gains Économiques du CETA', fontsize=12, fontweight='bold')
        ax.set_ylabel('Économies douanières (M€)', color='#0055A4')
        ax.tick_params(axis='y', labelcolor='#0055A4')
        ax.grid(True, alpha=0.3)
        
        # Croissance sectorielle en second axe
        ax2 = ax.twinx()
        ax2.plot(df['Annee'], df['Croissance_Sectorielle'], label='Croissance sectorielle (%)', 
                linewidth=2, color='#FF6600', alpha=0.8)
        ax2.set_ylabel('Croissance sectorielle (%)', color='#FF6600')
        ax2.tick_params(axis='y', labelcolor='#FF6600')
        
        # Combiner les légendes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    def _plot_before_after_comparison(self, df, ax):
        """Plot de comparaison avant/après CETA"""
        # Calculer les moyennes avant et après CETA
        before_ceta = df[df['Annee'] < 2017].mean()
        after_ceta = df[df['Annee'] >= 2017].mean()
        
        # Sélectionner les indicateurs à comparer
        indicators = ['Exportations_Vers_Canada', 'Importations_Du_Canada', 
                     'Creation_Emplois', 'Investissements_Etrangers']
        labels = ['Exportations', 'Importations', 'Emplois créés', 'Investissements']
        
        before_values = [before_ceta[ind] for ind in indicators]
        after_values = [after_ceta[ind] for ind in indicators]
        
        x = np.arange(len(indicators))
        width = 0.35
        
        ax.bar(x - width/2, before_values, width, label='Avant CETA (2014-2016)', color='#0055A4', alpha=0.7)
        ax.bar(x + width/2, after_values, width, label='Après CETA (2017-2027)', color='#FF0000', alpha=0.7)
        
        ax.set_title('Comparaison Avant/Après CETA', fontsize=12, fontweight='bold')
        ax.set_ylabel('Valeurs moyennes')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def _generate_ceta_insights(self, df):
        """Génère des insights analytiques sur le CETA"""
        print(f"🇪🇺🇨🇦 INSIGHTS ANALYTIQUES - Accord CETA - {self.country_sector}")
        print("=" * 70)
        
        # 1. Statistiques de base
        print("\n1. 📈 IMPACT COMMERCIAL:")
        export_growth = ((df['Exportations_Vers_Canada'].iloc[-1] / 
                         df['Exportations_Vers_Canada'].iloc[0]) - 1) * 100
        import_growth = ((df['Importations_Du_Canada'].iloc[-1] / 
                         df['Importations_Du_Canada'].iloc[0]) - 1) * 100
        avg_trade_balance = df['Balance_Commerciale'].mean()
        
        print(f"Croissance des exportations ({self.start_year}-{self.end_year}): {export_growth:.1f}%")
        print(f"Croissance des importations ({self.start_year}-{self.end_year}): {import_growth:.1f}%")
        print(f"Balance commerciale moyenne: {avg_trade_balance:.0f} M€")
        
        # 2. Impact économique
        print("\n2. 📊 IMPACT ÉCONOMIQUE:")
        total_jobs = df['Creation_Emplois'].sum()
        avg_gdp_impact = df['Impact_Sur_PIB'].mean() * 100
        total_savings = df['Economies_Douanieres'].sum()
        
        print(f"Emplois créés au total: {total_jobs:.0f}")
        print(f"Impact moyen sur le PIB: {avg_gdp_impact:.3f}%")
        print(f"Économies douanières totales: {total_savings:.0f} M€")
        
        # 3. Réduction des barrières
        print("\n3. 📋 RÉDUCTION DES BARRIÈRES:")
        tariff_reduction = ((df['Droits_Douane_Moyens'].iloc[0] - 
                           df['Droits_Douane_Moyens'].iloc[-1]) / 
                           df['Droits_Douane_Moyens'].iloc[0]) * 100
        ntb_reduction = ((df['Barrieres_Non_Tarifaires'].iloc[0] - 
                         df['Barrieres_Non_Tarifaires'].iloc[-1]) / 
                         df['Barrieres_Non_Tarifaires'].iloc[0]) * 100
        
        print(f"Réduction des droits de douane: {tariff_reduction:.1f}%")
        print(f"Réduction des barrières non tarifaires: {ntb_reduction:.1f}%")
        
        # 4. Spécificités du pays/secteur
        print(f"\n4. 🌟 SPÉCIFICITÉS DE {self.country_sector.upper()}:")
        print(f"Type: {self.config['type']}")
        if self.config["type"] == "pays_ue":
            print(f"Secteurs clés: {', '.join(self.config['secteurs_cles'])}")
        elif self.config["type"] == "secteur":
            print(f"Pays clés: {', '.join(self.config['pays_cles'])}")
        
        # 5. Événements marquants
        print("\n5. 📅 ÉVÉNEMENTS MARQUANTS:")
        print("• 2017: Entrée en vigueur provisoire du CETA")
        print("• 2017-2019: Augmentation progressive des échanges")
        print("• 2020: Impact de la pandémie COVID-19")
        print("• 2021-2022: Reprise post-COVID et ratification complète")
        print("• 2023-2027: Plein effet de l'accord et maturation des bénéfices")
        
        # 6. Recommandations stratégiques
        print("\n6. 💡 RECOMMANDATIONS STRATÉGIQUES:")
        if self.config["type"] == "pays_ue":
            print("• Maximiser les opportunités d'exportation dans les secteurs clés")
            print("• Adapter les normes et standards pour faciliter les échanges")
            print("• Renforcer la coopération réglementaire avec le Canada")
            print("• Développer des stratégies sectorielles ciblées")
        elif self.config["type"] == "secteur":
            print("• Identifier les niches de spécialisation dans la chaîne de valeur")
            print("• Développer des partenariats industriels transatlantiques")
            print("• Adapter les produits aux spécificités du marché canadien")
            print("• Profiter des reconnaissances mutuelles de qualifications")
        
        # Recommandations spécifiques selon les secteurs
        if "vin" in self.config.get("secteurs_cles", []):
            print("• Profiter de la protection des indications géographiques")
            print("• Développer le marketing des vins européens au Canada")
        if "fromage" in self.config.get("secteurs_cles", []):
            print("• Utiliser les quotas d'importation pour fromages fins")
            print("• Mettre en valeur les appellations d'origine protégée")
        if "automobile" in self.config.get("secteurs_cles", []):
            print("• Profiter de l'élimination des droits de douane")
            print("• Harmoniser les standards techniques pour réduire les coûts")
        if "services" in self.config.get("secteurs_cles", []):
            print("• Explorer les opportunités dans les services financiers")
            print("• Développer les services professionnels et techniques")

def main():
    """Fonction principale pour l'analyse de l'impact du CETA"""
    # Liste des pays et secteurs à analyser
    options = [
        "France", "Allemagne", "Italie", "Espagne", "Pays-Bas",
        "Canada", "UE-27", "Agriculture", "Automobile", "Services"
    ]
    
    print("🇪🇺🇨🇦 ANALYSE DE L'IMPACT DE L'ACCORD CETA UE-CANADA (2017-2027)")
    print("=" * 70)
    
    # Demander à l'utilisateur de choisir un pays/secteur
    print("Options disponibles:")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    
    try:
        choix = int(input("\nChoisissez le numéro du pays/secteur à analyser: "))
        if choix < 1 or choix > len(options):
            raise ValueError
        option_selectionnee = options[choix-1]
    except (ValueError, IndexError):
        print("Choix invalide. Sélection de la France par défaut.")
        option_selectionnee = "France"
    
    # Initialiser l'analyseur
    analyzer = CETAImpactAnalyzer(option_selectionnee)
    
    # Générer les données
    ceta_data = analyzer.generate_ceta_data()
    
    # Sauvegarder les données
    output_file = f'{option_selectionnee}_ceta_data_2017_2027.csv'
    ceta_data.to_csv(output_file, index=False)
    print(f"💾 Données sauvegardées: {output_file}")
    
    # Aperçu des données
    print("\n👀 Aperçu des données:")
    print(ceta_data[['Annee', 'Exportations_Vers_Canada', 'Importations_Du_Canada', 
                    'Balance_Commerciale', 'Creation_Emplois']].head())
    
    # Créer l'analyse
    print("\n📈 Création de l'analyse CETA...")
    analyzer.create_ceta_analysis(ceta_data)
    
    print(f"\n✅ Analyse CETA pour {option_selectionnee} terminée!")
    print(f"📊 Période: {analyzer.start_year}-{analyzer.end_year}")
    print("📦 Données: Échanges commerciaux, emploi, investissements, impacts économiques")

if __name__ == "__main__":
    main()