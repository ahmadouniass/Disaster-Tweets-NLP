from pathlib import Path

import pandas as pd


def find_best_models(output_dir="outputs"):
    """
    Scanne le dossier outputs pour trouver tous les fichiers de résultats
    et identifier les meilleurs modèles.
    """
    path = Path(output_dir)
    all_results = []

    # On cherche tous les fichiers qui finissent par results.csv
    files = list(path.glob("**/NB*_results.csv")) + \
            list(path.glob("**/*tuned_results.csv")) + \
            list(path.glob("**/*baseline_results.csv"))

    print(f"🔍 Analyse de {len(files)} fichiers de résultats...\n")

    for file in files:
        try:
            df = pd.read_csv(file)
            # On vérifie que les colonnes nécessaires existent
            required_cols = ['pipeline', 'test_f1_class_1']
            if all(col in df.columns for col in required_cols):
                # Si le nom est "test", on utilise le nom du dossier parent
                df.loc[df['pipeline'] == 'test', 'pipeline'] = f"{file.parent.name}_tuned"

                cols_to_keep = [
                    'pipeline', 'test_f1_class_1', 'test_recall_class_1',
                    'test_precision_class_1', 'test_accuracy'
                ]
                # Filtrer les colonnes qui existent réellement dans ce fichier
                existing_cols = [c for c in cols_to_keep if c in df.columns]
                all_results.append(df[existing_cols])
        except Exception as e:
            print(f"⚠️ Erreur lors de la lecture de {file.name}: {e}")

    if not all_results:
        print("❌ Aucun résultat valide trouvé.")
        return

    # Fusion de tous les résultats
    results_df = pd.concat(all_results, ignore_index=True)

    # On renomme "test" par quelque chose de plus explicite
    results_df['pipeline'] = results_df['pipeline'].replace({'test': '🏆 DistilBERT_Tuned (Champion)'})

    # Suppression des doublons potentiels
    results_df = results_df.drop_duplicates(subset=['pipeline'])

    # Tri par F1-Score décroissant
    results_df = results_df.sort_values(by='test_f1_class_1', ascending=False).reset_index(drop=True)

    # Affichage du podium
    print("🏆 --- LE PODIUM DES MODÈLES (F1-SCORE CLASSE 1) --- 🏆")
    print(results_df.head(10).to_string(index=True))

    champion = results_df.iloc[0]
    print(f"\n✨ LE CHAMPION EST : {champion['pipeline']}")
    print(f"📈 F1-Score : {champion['test_f1_class_1']:.4f}")
    print(f"🎯 Précision : {champion['test_precision_class_1']:.4f}")
    print(f"📢 Rappel    : {champion['test_recall_class_1']:.4f}")
    print(f"✅ Accuracy  : {champion['test_accuracy']:.4f}")

if __name__ == "__main__":
    find_best_models()
