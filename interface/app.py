"""
🌾 Application de Recommandation de Cultures
Fichier: app.py (Version Corrigée)
"""

import gradio as gr
import pickle
import numpy as np

# ========================================
# CHARGEMENT DU MODÈLE
# ========================================

try:
    model = pickle.load(open('model.pkl', 'rb'))
    minmax_scaler = pickle.load(open('minmaxscaler.pkl', 'rb'))
    
    # Vérification du type de standard_scaler
    try:
        standard_scaler = pickle.load(open('standscaler.pkl', 'rb'))
        
        # Si c'est un RandomForestClassifier, on ne l'utilise pas
        if hasattr(standard_scaler, 'n_estimators'):
            print("⚠️ Warning: standscaler.pkl contient un RandomForestClassifier!")
            print("→ Utilisation uniquement de MinMaxScaler")
            standard_scaler = None
    except:
        standard_scaler = None
        
except Exception as e:
    print(f"❌ Erreur de chargement: {e}")
    raise

# ========================================
# DICTIONNAIRE DES CULTURES
# ========================================

CULTURES = {
    1: '🌾 Riz', 2: '🌽 Maïs', 3: '🌿 Jute', 4: '☁️ Coton',
    5: '🥥 Noix de coco', 6: '🥭 Papaye', 7: '🍊 Orange', 8: '🍎 Pomme',
    9: '🍈 Melon', 10: '🍉 Pastèque', 11: '🍇 Raisins', 12: '🥭 Mangue',
    13: '🍌 Banane', 14: '🍑 Grenade', 15: '🫘 Lentille', 16: '🫘 Haricot noir',
    17: '🫘 Haricot mungo', 18: '🫘 Haricot papillon', 19: '🫘 Pois d\'Angole',
    20: '🫘 Haricots rouges', 21: '🫘 Pois chiche', 22: '☕ Café'
}

# ========================================
# FONCTION DE PRÉDICTION
# ========================================

def predire_culture(N, P, K, temperature, humidity, ph, rainfall):
    """Prédit la meilleure culture selon les paramètres"""
    
    try:
        # Préparer les données
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        
        # Normalisation avec MinMaxScaler uniquement
        features_normalized = minmax_scaler.transform(features)
        
        # Si standard_scaler existe ET est valide, on l'applique
        if standard_scaler is not None and hasattr(standard_scaler, 'transform'):
            features_normalized = standard_scaler.transform(features_normalized)
        
        # Prédiction
        prediction = model.predict(features_normalized)[0]
        culture = CULTURES.get(int(prediction), "Culture inconnue")
        
        # Message de résultat
        resultat = f"""
# 🎯 Recommandation

## Culture idéale : {culture}

### 📊 Vos paramètres :
- **Azote (N)** : {N}
- **Phosphore (P)** : {P}
- **Potassium (K)** : {K}
- **Température** : {temperature}°C
- **Humidité** : {humidity}%
- **pH** : {ph}
- **Pluviométrie** : {rainfall} mm

### ✅ Cette culture est optimale pour votre sol !
        """
        
        return resultat
        
    except Exception as e:
        return f"❌ Erreur lors de la prédiction : {str(e)}"

# ========================================
# INTERFACE GRADIO
# ========================================

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="green"),
    css="""
        .output-markdown {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
            padding: 25px;
            border-radius: 15px;
        }
    """
) as demo:
    
    # Titre
    gr.Markdown("""
    # 🌾 Système de Recommandation de Cultures
    ### Trouvez la culture idéale pour votre terrain
    """)
    
    # Deux colonnes
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🧪 Composition du Sol")
            N = gr.Slider(0, 140, 50, label="💚 Azote (N)")
            P = gr.Slider(5, 145, 53, label="🟠 Phosphore (P)")
            K = gr.Slider(5, 205, 48, label="🔵 Potassium (K)")
            ph = gr.Slider(3.5, 10, 6.5, step=0.1, label="⚗️ pH du Sol")
        
        with gr.Column():
            gr.Markdown("### 🌦️ Conditions Climatiques")
            temp = gr.Slider(8, 44, 25, step=0.1, label="🌡️ Température (°C)")
            humidity = gr.Slider(14, 100, 71, step=0.1, label="💧 Humidité (%)")
            rainfall = gr.Slider(20, 300, 103, step=0.1, label="🌧️ Pluie (mm)")
    
    # Boutons
    with gr.Row():
        btn_predict = gr.Button("🔍 Recommander", variant="primary", size="lg")
        btn_clear = gr.Button("🔄 Effacer", variant="secondary")
    
    # Résultat
    output = gr.Markdown(elem_classes="output-markdown")
    
    # Exemples
    gr.Examples(
        examples=[
            [90, 42, 43, 20.8, 82, 6.5, 203],  # Riz
            [20, 67, 20, 26, 52, 5.9, 60],     # Maïs
            [80, 40, 40, 20, 80, 6.5, 200],    # Banane
        ],
        inputs=[N, P, K, temp, humidity, ph, rainfall],
    )
    
    # Actions
    btn_predict.click(
        predire_culture,
        inputs=[N, P, K, temp, humidity, ph, rainfall],
        outputs=output
    )
    
    btn_clear.click(
        lambda: (50, 53, 48, 25, 71, 6.5, 103, ""),
        outputs=[N, P, K, temp, humidity, ph, rainfall, output]
    )
    
    # Pied de page
    gr.Markdown("""
    ---
    **ℹ️ Info** : Modèle avec précision de 99.3% | 22 cultures disponibles
    """)

# ========================================
# LANCEMENT
# ========================================

if __name__ == "__main__":
    demo.launch(share=True)