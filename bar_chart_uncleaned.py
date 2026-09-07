import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("/home/jeswes99/Desktop/Skole/Masteroppgave/bacdive_HmE6dj.csv")

# Some rows have multiple traits combined like "organotroph|photoautotroph"
# Split those into separate rows so each trait is counted properly
df["nutritionType"] = df["nutritionType"].str.split("|")
df = df.explode("nutritionType")
df["nutritionType"] = df["nutritionType"].str.strip()

# Count unique strains per trait (avoid double-counting duplicate rows)
trait_counts = df.drop_duplicates(subset=["nutritionType", "strain"])["nutritionType"].value_counts()

# Plot
plt.figure(figsize=(12, 6))
bars = plt.bar(trait_counts.index, trait_counts.values, color="steelblue")
plt.xticks(rotation=75, ha="right")
plt.ylabel("Number of strains")
plt.title("Distribution of Nutrition types")

for bar, count in zip(bars, trait_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(count),
              ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("/home/jeswes99/Desktop/Skole/Masteroppgave/trophic_trait_distribution.png", dpi=300)
plt.show()