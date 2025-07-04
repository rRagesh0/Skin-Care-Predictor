import pandas as pd

skin_type_concerns = {
    'dry acne': {
        'directly_related': [
            'Acne or Blemishes', 'Excess Oil', 'Blackheads and Whiteheads', 
            'Pore Care', 'Oil Control', 'Anti Acne Scarring'
        ],
        'indirectly_related': ['Deep Cleansing', 'Hydration']
    },
    'dry redness': {
        'directly_related': ['Skin Inflammation', 'Uneven Texture'],
        'indirectly_related': ['Hydration', 'Anti-Pollution', 'Sun Protection', 'Softening and Smoothening']
    },
    'dry wrinkle': {
        'directly_related': ['Anti-Ageing', 'Deep Nourishment', 'Softening and Smoothening'],
        'indirectly_related': ['Hydration', 'Brightening']
    },
    'dry darkcircle': {
        'directly_related': ['Dark Spots', 'Brightening', 'Pigmentation', 'Dullness'],
        'indirectly_related': ['Hydration', 'Anti-Pollution', 'Sun Protection']
    },
    'normal acne': {
        'directly_related': [
            'Acne or Blemishes', 'Excess Oil', 'Blackheads and Whiteheads', 
            'Pore Care', 'Oil Control', 'Anti Acne Scarring'
        ],
        'indirectly_related': ['Deep Cleansing', 'Hydration']
    },
    'normal redness': {
        'directly_related': ['Skin Inflammation', 'Uneven Texture'],
        'indirectly_related': ['Hydration', 'Anti-Pollution', 'Sun Protection', 'Softening and Smoothening']
    },
    'normal wrinkle': {
        'directly_related': ['Anti-Ageing', 'Deep Nourishment', 'Softening and Smoothening'],
        'indirectly_related': ['Hydration', 'Brightening']
    },
    'normal darkcircle': {
        'directly_related': ['Dark Spots', 'Brightening', 'Pigmentation', 'Dullness'],
        'indirectly_related': ['Hydration', 'Anti-Pollution', 'Sun Protection']
    },
    'oily acne': {
        'directly_related': [
            'Acne or Blemishes', 'Excess Oil', 'Blackheads and Whiteheads', 
            'Pore Care', 'Oil Control', 'Anti Acne Scarring'
        ],
        'indirectly_related': ['Deep Cleansing', 'Hydration']
    },
    'oily redness': {
        'directly_related': ['Skin Inflammation', 'Uneven Texture'],
        'indirectly_related': ['Hydration', 'Anti-Pollution', 'Sun Protection', 'Softening and Smoothening']
    },
    'oily wrinkle': {
        'directly_related': ['Anti-Ageing', 'Deep Nourishment', 'Softening and Smoothening'],
        'indirectly_related': ['Hydration', 'Brightening']
    },
    'oily darkcircle': {
        'directly_related': ['Dark Spots', 'Brightening', 'Pigmentation', 'Dullness'],
        'indirectly_related': ['Hydration', 'Anti-Pollution', 'Sun Protection']
    }
}

def get_price_ranges(filtered_df):
    if not filtered_df.empty:
        filtered_df = filtered_df.dropna(subset=['price'])
        if len(filtered_df) > 0:
            low_threshold = filtered_df['price'].quantile(0.25)
            high_threshold = filtered_df['price'].quantile(0.75)
            return low_threshold, high_threshold
        else:
            return 0, 0
    else:
        return 0, 0

def recommender(skin_type, skin_condition):
    try:
        df = pd.read_excel('condition_based_scraper_filtered.xlsx')
        print("Data loaded successfully!")
        df.columns = df.columns.str.strip()
        df['skin type'] = df['skin type'].str.strip().str.lower()
        df['concern'] = df['concern'].str.strip().str.lower()
        
        skin_type = skin_type.lower()
        skin_condition = skin_condition.lower()
        key = f"{skin_type} {skin_condition}"

        if key in skin_type_concerns:
            concerns = skin_type_concerns[key]
            print(f"Directly related concerns: {concerns['directly_related']}")
            print(f"Indirectly related concerns: {concerns['indirectly_related']}")

            filtered_df = df[df['skin type'].str.contains(skin_type, case=False, na=False)]
            matching_products = filtered_df[filtered_df['concern'].str.contains('|'.join(concerns['directly_related']), case=False, na=False)]

            if matching_products.empty:
                matching_products = filtered_df[filtered_df['concern'].str.contains('|'.join(concerns['indirectly_related']), case=False, na=False)]
                print(f"Found {matching_products.shape[0]} products matching indirect concerns.")

            if not matching_products.empty:
                low_threshold, high_threshold = get_price_ranges(matching_products)
                low_products = matching_products[matching_products['price'] <= low_threshold].head(2)
                medium_products = matching_products[(matching_products['price'] > low_threshold) & (matching_products['price'] < high_threshold)].head(2)
                high_products = matching_products[matching_products['price'] >= high_threshold].head(2)

                # Combine all products
                recommended_products = pd.concat([low_products, medium_products, high_products])
                return recommended_products.to_dict(orient="records")
            else:
                print("No matching products found.")
                return None
        else:
            print(f"Key {key} not found in skin_type_concerns.")
            return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
