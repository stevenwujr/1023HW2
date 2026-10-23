/**
 * Dataset Configurations for Multiple Linear Regression (CRISP-DM)
 * Strictly adheres to 13 < Features < 40 (Kaggle Datasets with 14-39 features)
 */

window.DATASET_CONFIGS = {
  flood: {
    id: 'flood',
    label: '🌊 Kaggle Problem 1: 洪水機率預測 (Flood Probability - 20 Features, S4E5)',
    file: './data/flood_prediction_sample.csv',
    title: 'Kaggle Playground Series s4e5: 洪水機率預測',
    subtitle: '利用 20 項氣象、水利管理、森林砍伐與基建老化特徵預測區域洪澇機率',
    featuresCount: 20,
    target: 'FloodProbability',
    targetLabel: '洪水機率 (Flood Probability)',
    targetUnit: '機率值 [0.28 ~ 0.75]',
    kaggleUrl: 'https://www.kaggle.com/competitions/playground-series-s4e5',
    suitability: '極佳 (20 個連續數值特徵，非常適合 OLS、Ridge、Lasso、殘差分析與五大假設檢定)',
    description: '洪澇災害是全球最具破壞性的自然災害之一。本任務選用 Kaggle Playground Series s4e5 公開資料集，利用 20 項連續數值特徵量化氣候變遷與人為環境對洪水風險的邊際貢獻。'
  },
  istanbul: {
    id: 'istanbul',
    label: '🏢 Kaggle Problem 2: 伊斯坦堡公寓房價 (Istanbul Apartment Prices - 18 Features)',
    file: './data/istanbul_apartment_sample.csv',
    title: 'Kaggle: 伊斯坦堡公寓售價多元線性回歸預測',
    subtitle: '利用 18 項房屋結構、樓層、地鐵距離、市中心距離與學區指標預測房屋售價',
    featuresCount: 18,
    target: 'Price',
    targetLabel: '公寓售價 (Apartment Price)',
    targetUnit: '千美元 (Thousand USD)',
    kaggleUrl: 'https://www.kaggle.com/datasets/brahimenesulusoy/istanbul-apartment-prices-2026',
    suitability: '良好 (18 個特徵，適合特徵篩選、共線性排查與信賴/預測區間推導)',
    description: '伊斯坦堡橫跨歐亞兩洲，房產價格受到坪數、屋齡、距離地鐵站距離、物業保全及周邊綠覆率等多重變數影響。本任務建立多元線性回歸模型協助房產估價與投資風險審核。'
  },
  life: {
    id: 'life',
    label: '🏥 Kaggle Problem 3: WHO 預期壽命預測 (WHO Life Expectancy - 20 Features)',
    file: './data/life_expectancy_sample.csv',
    title: 'Kaggle: 世界衛生組織 (WHO) 國家預期壽命預測',
    subtitle: '利用 20 項醫療衛生、疫苗覆蓋率、成人死亡率與 GDP 經濟指標預測平均壽命',
    featuresCount: 20,
    target: 'Life_expectancy',
    targetLabel: '平均預期壽命 (Life Expectancy)',
    targetUnit: '歲 (Years)',
    kaggleUrl: 'https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who',
    suitability: '極佳 (20 個連續特徵，醫療與社會經濟多元變數回歸)',
    description: '由世界衛生組織 (WHO) 整合之全球健康資料集，涵蓋成人死亡率、嬰兒死亡、肝炎/小兒麻痺/白喉疫苗接種率、醫療衛生支出佔比與受教育年限等 20 項指標。'
  }
};
