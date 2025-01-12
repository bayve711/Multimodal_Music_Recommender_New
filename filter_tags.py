import pandas as pd
import ast
import os
from collections import Counter

#collect all genres
id_genres_df = pd.read_csv("data\id_genres_mmsr.tsv", sep="\t")

#flatten into uniq set
genre_set = set()
for genres in id_genres_df['genre']:
    genres_list = ast.literal_eval(genres)  # Convert string to list
    genre_set.update(genres_list)

#manual genre inclusion
additional_genres = {"alternative", "indie", "electronic", "rnb", "punk rock", "heavy metal"}
genre_set.update(additional_genres)

print(f"Collected genres (including additional): {genre_set}")

#load tags and filter genres out
id_tags_df = pd.read_csv("data\id_tags_dict.tsv", sep="\t")

#dict createt
filtered_tags = []
tag_counter = Counter()
for tags_weight in id_tags_df['(tag, weight)']:
    tags_dict = ast.literal_eval(tags_weight)  #string to dict
    filtered_tags_dict = {tag: weight for tag, weight in tags_dict.items() if tag not in genre_set}
    filtered_tags.append(filtered_tags_dict)
    tag_counter.update(filtered_tags_dict)

#new column
id_tags_df['filtered_tags'] = filtered_tags


output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

#save
output_file = os.path.join(output_dir, "filtered_id_tags_dict.tsv")
id_tags_df.to_csv(output_file, sep="\t", index=False, encoding='utf-8')

print(f"Filtered tags saved to '{output_file}'")

#50 most common tags
top_50_tags = tag_counter.most_common(50)
print("Top 50 most common tags:")
for tag, count in top_50_tags:
    print(f"{tag}: {count}")
