import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
import os
import re # Import the re module for regular expressions

# For Google Gemini
import google.generativeai as genai

# For Text-to-Speech
import pyttsx3
import threading # Used to run TTS in a separate thread to avoid blocking Streamlit


# Set device globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Title - Set layout to "wide" for more horizontal space
st.set_page_config(page_title="Crop Disease Detection", layout="wide")
st.title("🌿 Crop Disease Detection App")
st.markdown("Select your crop, upload or capture an image, and get disease prediction with remedies.")

# Crop selection
crop = st.selectbox("Choose your crop", ["Grapes", "Potato", "Peanut", "Tomato", "Rice", "Cashew"]) # Added Cashew

# --- START ENRICHED MODEL_INFO DICTIONARY ---
model_info = {
    "Grapes": {
        "model_path": "resnet50_grapes.pt",
        "labels": ["Black Rot", "ESCA", "Healthy", "Leaf Blight"],
        "num_classes": 4,
        "fc_type": "single_linear",
        "remedies": {
            "Black Rot": {
                "cause": "Caused by the fungus *Guignardia bidwellii*, which overwinters in mummified berries and cane lesions.",
                "severity": "High: Can lead to significant crop loss (up to 80%) if untreated, affecting leaves, stems, and fruit.",
                "symptoms": "Circular, tan-brown spots with dark borders on leaves; black, shriveled berries ('mummies'); elongated lesions on stems.",
                "conditions_favoring": "Warm (20-27°C) and wet conditions (6+ hours of leaf wetness) during spring and early summer.",
                "disease_cycle": "Spores are released from overwintering sites, splashed onto new growth by rain, infect, and then produce more spores to reinfect.",
                "impact": "Reduced yield, poor fruit quality, un marketable berries due to rot.",
                "season": "Early to mid-monsoon (June-July in India), extending through harvest.",
                "organic_options": ["Copper-based fungicides (e.g., Bordeaux mixture), Neem oil (for early stages).", "Improve air circulation through pruning.", "Remove infected plant parts."],
                "chemical_options": ["Fungicides like Mancozeb, Myclobutanil, or Captan. Apply protectants before infection and systemic post-infection."],
                "recommended_varieties": ["Concord (moderate resistance), some Vitis vinifera cultivars show varying susceptibility."],
                "next_steps": [
                    "Inspect vines regularly for early signs.",
                    "Ensure proper spacing and pruning for air circulation.",
                    "Apply recommended fungicides preventatively or as directed.",
                    "Sanitize vineyard by removing mummified berries and infected canes."
                ],
                "external_links": {
                    "Cornell Extension on Grape Black Rot": "https://grapes.extension.org/grape-black-rot/",
                    "UGA Extension - Grape Diseases": "https://extension.uga.edu/publications/detail.html?number=C884&title=Diseases%20of%20Grape%20in%20Georgia"
                }
            },
            "ESCA": {
                "cause": "A complex of fungal pathogens (*Phaeomoniella chlamydospora*, *Phaeoacremonium minimum*, and others) that infect the woody parts of the vine, primarily entering through pruning wounds.",
                "severity": "Medium to High: Can cause vine decline and eventual death over several years, especially in older vines. Affects vine longevity.",
                "symptoms": "Characteristic 'tiger-stripe' necrosis on leaves (interveinal yellowing/reddening), stunted shoots, internal wood discoloration (dark streaks or rot).",
                "conditions_favoring": "Humid conditions after pruning, allowing fungal spores to enter wounds. Stressed vines are more susceptible.",
                "disease_cycle": "Fungi colonize the woody tissues, leading to wood decay and toxin production. Spores are released during wet periods.",
                "impact": "Gradual decline of vine productivity, reduced yield, and eventual vine death, requiring costly replanting.",
                "season": "Symptoms typically appear in warm summer months (July-August in India). Infections occur year-round, especially after pruning.",
                "organic_options": ["Seal pruning wounds immediately with natural wound protectants (e.g., bio-fungicides like *Trichoderma* spp.).", "Prune during dry periods."],
                "chemical_options": ["Few effective chemical cures. Focus on prevention with wound protectants containing fungicides (e.g., thiophanate-methyl, cyproconazole)."],
                "recommended_varieties": ["No truly resistant varieties, but some show more tolerance than others. Focus on cultural practices."],
                "next_steps": [
                    "Prune only during dry periods to minimize infection risk.",
                    "Apply wound protectant immediately after pruning cuts.",
                    "Monitor vines for internal wood discoloration during pruning.",
                    "Remove and destroy severely affected vines to prevent spread."
                ],
                "external_links": {
                    "UC Davis IPM on Grape Trunk Diseases": "https://www.ipm.ucdavis.edu/agriculture/grape/trunk-diseases/",
                    "USDA ARS - Grapevine Trunk Diseases": "https://www.ars.usda.gov/southeast-area/fayetteville-ar/grape-genetics-environmental-physiology-research/docs/grapevine-trunk-diseases/"
                }
            },
            "Healthy": {
                "cause": "N/A",
                "severity": "N/A",
                "symptoms": "Vibrant green leaves, robust growth, no visible spots, discolorations, or deformities. Strong fruit set.",
                "conditions_favoring": "Optimal temperature, light, water, and nutrient availability.",
                "disease_cycle": "N/A",
                "impact": "Maximum yield and high-quality fruit.",
                "season": "All growing seasons with proper care.",
                "organic_options": ["Balanced organic fertilizers, compost application, biological pest control."],
                "chemical_options": ["Balanced synthetic fertilizers, pest monitoring, targeted conventional sprays only if necessary."],
                "recommended_varieties": ["All healthy varieties."],
                "next_steps": [
                    "Continue balanced fertilization and irrigation.",
                    "Regularly monitor for any signs of pests or diseases.",
                    "Ensure proper canopy management for air circulation and light penetration.",
                    "Prune annually for vine vigor and yield."
                ],
                "external_links": {
                    "FAO - Grape Production Guide": "https://www.fao.org/land-water/events/documents/detail/en/c/1188373/",
                    "ICAR-NRC for Grapes (India)": "https://nrcgrapes.icar.gov.in/"
                }
            },
            "Leaf Blight": {
                "cause": "Often caused by various fungal pathogens (e.g., *Alternaria* spp., *Cercospora* spp.) or bacterial infections, commonly found in humid environments.",
                "severity": "Medium: Can cause defoliation, reducing photosynthesis and fruit quality, but rarely lethal to the entire vine.",
                "symptoms": "Irregular, water-soaked spots on leaves that turn brown/black, often with a yellow halo. Leaves may curl, dry up, and drop prematurely.",
                "conditions_favoring": "High humidity, prolonged leaf wetness, and moderate temperatures (20-30°C).",
                "disease_cycle": "Spores spread by wind and splashing rain, infecting leaves. Can survive on plant debris.",
                "impact": "Reduced vine vigor, lower fruit quality, and potential yield loss due to reduced photosynthetic capacity.",
                "season": "Monsoon season (July-September in India) and humid periods.",
                "organic_options": ["Copper-based fungicides, Neem oil, removal of infected leaves and plant debris.", "Improve air circulation."],
                "chemical_options": ["Fungicides containing chlorothalonil, mancozeb, or strobilurins (e.g., azoxystrobin)."],
                "recommended_varieties": ["Some varieties show less susceptibility; local agricultural extension can advise on region-specific resistant types."],
                "next_steps": [
                    "Remove and destroy severely affected leaves and plant debris.",
                    "Ensure proper spacing and pruning for better air circulation.",
                    "Avoid overhead irrigation; use drip irrigation instead.",
                    "Apply recommended fungicides at appropriate intervals, especially during humid periods."
                ],
                "external_links": {
                    "Missouri Extension - Grape Disease Control": "https://ipm.mo.gov/MEG/2012/3/Managing-Grape-Diseases-Part-1-Early-Season-Diseases/",
                    "Purdue Extension - Grape Leaf Diseases": "https://www.extension.purdue.edu/extmedia/BP/BP-68.html"
                }
            }
        }
    },
    "Potato": {
        "model_path": "resnet50_potato.pt",
        "labels": ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"],
        "num_classes": 3,
        "fc_type": "single_linear",
        "remedies": {
            "Potato___Early_blight": {
                "cause": "Caused by the fungus *Alternaria solani*, which overwinters in infected plant debris and soil. Spores are spread by wind and rain.",
                "severity": "Medium: Can cause significant yield reduction (up to 20-30%) if not controlled, primarily affecting older leaves and reducing tuber size.",
                "symptoms": "Dark brown, concentric rings (target-like spots) on older leaves. Lesions may be surrounded by a yellow halo. Leaves may yellow and drop prematurely.",
                "conditions_favoring": "Warmer temperatures (20-30°C), prolonged leaf wetness from dew or rain, and high humidity. Often appears after flowering.",
                "disease_cycle": "Spores infect leaves, develop lesions, and produce more spores. Can also infect tubers.",
                "impact": "Reduced tuber size and yield due to premature defoliation.",
                "season": "Mid to late growing season (August-October in India, depending on planting time).",
                "organic_options": ["Copper-based fungicides (e.g., Bordeaux mixture), use of biological control agents (e.g., *Bacillus subtilis*).", "Crop rotation."],
                "chemical_options": ["Fungicides containing chlorothalonil, azoxystrobin, difenoconazole. Apply protectants before symptoms appear."],
                "recommended_varieties": ["Kufri Chipsona-1, Kufri Jyoti (show some tolerance), but resistance varies by region. Consult local extension."],
                "next_steps": [
                    "Practice 2-3 year crop rotation.",
                    "Ensure proper plant spacing for air circulation.",
                    "Maintain balanced fertilization, avoiding excessive nitrogen.",
                    "Apply protective fungicides according to schedule."
                ],
                "external_links": {
                    "Purdue Extension - Potato Early Blight": "https://www.extension.purdue.edu/extmedia/BP/BP-130-W.pdf",
                    "ICAR-CPRI - Potato Diseases (India)": "https://cpri.icar.gov.in/Potato_Diseases_Insect_Pests.html"
                }
            },
            "Potato___Late_blight": {
                "cause": "Caused by the oomycete (*Phytophthora infestans*), a highly aggressive pathogen. Spread rapidly by wind and rain, especially from infected seed potatoes or volunteer plants.",
                "severity": "High: Rapidly spreads and can devastate entire fields within days if conditions are favorable and left untreated, leading to complete crop loss.",
                "symptoms": "Water-soaked lesions on leaves and stems that quickly turn brown/black, often with a fuzzy white mold on the underside of leaves in humid conditions. Brown rot in tubers.",
                "conditions_favoring": "Cool (10-20°C), wet, and very humid conditions (relative humidity >90%, 10+ hours leaf wetness). '7-70' rule (7°C min, 70% RH min).",
                "disease_cycle": "Spores produced rapidly, spread by wind, infect leaves/stems/tubers. Survives in infected tubers.",
                "impact": "Complete destruction of foliage and rotting of tubers, rendering the crop unmarketable.",
                "season": "Cool and wet periods, typically late growing season or monsoon (August-November in India).",
                "organic_options": ["Copper-based fungicides (e.g., Bordeaux mixture), bio-fungicides (e.g., *Bacillus amyloliquefaciens*).", "Ensure good field drainage.", "Destroy infected plant debris."],
                "chemical_options": ["Systemic fungicides (e.g., Ridomil Gold, Revus, Infinito) are crucial. Apply preventatively during high-risk periods."],
                "recommended_varieties": ["Kufri Badshah, Kufri Himalini (show resistance), Kufri Pukhraj (some tolerance). Consult local extension for resistant varieties."],
                "next_steps": [
                    "Use certified disease-free seed potatoes.",
                    "Apply systemic fungicides preventatively in high-risk periods.",
                    "Monitor weather conditions closely; act immediately when conditions are favorable.",
                    "Ensure proper hilling to protect tubers from spores."
                ],
                "external_links": {
                    "Cornell Extension - Late Blight of Potato": "https://blogs.cornell.edu/plantpath/extension/vegetables/late-blight/",
                    "Potato Late Blight Forecast (India)": "https://www.agrimet.tn.nic.in/PotatoLateBlightForecast.html" # Example of relevant local info
                }
            },
            "Potato___healthy": {
                "cause": "N/A",
                "severity": "N/A",
                "symptoms": "Lush green foliage, strong stems, and uniform growth. Absence of spots, lesions, or discolored areas.",
                "conditions_favoring": "Optimal soil moisture, balanced nutrients, adequate sunlight, and moderate temperatures.",
                "disease_cycle": "N/A",
                "impact": "Achieving maximum potential yield and high-quality tubers.",
                "season": "All growing seasons with optimal cultural practices.",
                "organic_options": ["Compost application, crop rotation, companion planting, biological pest control."],
                "chemical_options": ["Balanced NPK fertilizers, targeted pest control if needed."],
                "recommended_varieties": ["All healthy varieties."],
                "next_steps": [
                    "Maintain balanced fertilization based on soil tests.",
                    "Ensure consistent, adequate watering.",
                    "Practice good hilling to protect tubers from light and pests.",
                    "Regularly scout for early signs of disease or pests."
                ],
                "external_links": {
                    "FAO - Potato Production Guide": "https://www.fao.org/potato-2008/production/en/",
                    "ICAR-CPRI (Indian Potato Research)": "https://cpri.icar.gov.in/"
                }
            }
        }
    },
    "Peanut": {
        "model_path": "resnet50_peanut.pt",
        "labels": ["Peanut_rust", "Peanut_nutrition_deficiency", "Peanut_leaf_spot", "Peanut_healthy_leaf"],
        "num_classes": 4,
        "fc_type": "custom_sequential_peanut",
        "remedies": {
            "Peanut_rust": {
                "cause": "Caused by the fungus *Puccinia arachidis*. Spores are wind-borne and thrive in warm, humid conditions.",
                "severity": "High: Can cause severe defoliation (up to 70-80%) and significant yield losses, especially if infection occurs early.",
                "symptoms": "Orange-reddish pustules (rust spores) on the underside of leaves, which may rupture to release powdery spores. Upper leaf surface shows corresponding yellow spots.",
                "conditions_favoring": "Warm temperatures (25-30°C) and high humidity (>95% RH) or prolonged leaf wetness.",
                "disease_cycle": "Spores land on leaves, germinate, infect, and produce new pustules. Can spread rapidly within a field.",
                "impact": "Reduced pod fill, decreased kernel quality, and significant yield reductions.",
                "season": "Often prevalent during the rainy season (monsoon, July-September in India) or late growing season.",
                "organic_options": ["Neem oil sprays, removal of infected crop residue, use of biological fungicides (e.g., *Pseudomonas fluorescens*)."],
                "chemical_options": ["Fungicides like tebuconazole, propiconazole, azoxystrobin. Apply preventatively or at early signs."],
                "recommended_varieties": ["Some modern varieties like ICGV 91114, JL 24 (with good management) show tolerance. Consult local research institutes."],
                "next_steps": [
                    "Scout fields regularly, especially during humid periods.",
                    "Apply recommended fungicides preventatively or at first signs.",
                    "Ensure good plant spacing for air circulation.",
                    "Destroy crop residue and volunteer peanuts to reduce inoculum."
                ],
                "external_links": {
                    "ICRISAT - Peanut Rust": "https://www.icrisat.org/Disease/Peanut_Rust.htm",
                    "TNAU Agritech Portal - Groundnut Diseases": "https://agritech.tnau.ac.in/agriculture/agri_major_crops_groundnut_diseases.html"
                }
            },
            "Peanut_nutrition_deficiency": {
                "cause": "Lack of essential macro (N, P, K) or micronutrients (Fe, Zn, Mg, Mn, B) in the soil, nutrient imbalance, or soil pH issues hindering uptake.",
                "severity": "Varies (Low to Medium): Depending on the specific nutrient and severity of deficiency, plant growth, yield, and kernel quality can be significantly impacted.",
                "symptoms": "Symptoms vary by nutrient: e.g., yellowing between veins (iron/magnesium), stunted growth, purplish discoloration (phosphorus), distorted leaves (boron), pale green leaves (nitrogen).",
                "conditions_favoring": "Poor soil fertility, imbalanced fertilizer application, compacted soil, extreme soil pH (too high or too low).",
                "disease_cycle": "N/A (physiological disorder)",
                "impact": "Reduced plant vigor, lower yields, poor kernel development, and decreased market value.",
                "season": "Can occur throughout the growing season if soil conditions are suboptimal.",
                "organic_options": ["Apply well-decomposed compost or farmyard manure.", "Use organic nutrient sources (e.g., bone meal for P, kelp meal for micros).", "Soil test and adjust pH with natural amendments."],
                "chemical_options": ["Apply appropriate synthetic fertilizers based on soil test results (e.g., chelated micronutrients for foliar application)."],
                "recommended_varieties": ["N/A (issue is soil-based, not varietal resistance)."],
                "next_steps": [
                    "Conduct a comprehensive soil test to identify specific deficiencies.",
                    "Apply recommended fertilizers or micronutrient mixes based on test results.",
                    "Ensure proper irrigation to facilitate nutrient uptake.",
                    "Adjust soil pH if necessary to optimize nutrient availability."
                ],
                "external_links": {
                    "FAO - Peanut Production Guide": "https://www.fao.org/docrep/008/ad227e/ad227e05.htm",
                    "ICAR - Nutrient Management in Groundnut": "https://www.icar.org.in/sites/default/files/Groundnut-Production-Technologies.pdf"
                }
            },
            "Peanut_leaf_spot": {
                "cause": "Primarily caused by fungal pathogens *Cercospora arachidicola* (early leaf spot) or *Cercosporidium personatum* (late leaf spot). Spores are wind-borne.",
                "severity": "Medium to High: Can lead to premature defoliation, significantly reducing photosynthesis, resulting in reduced yield and quality (up to 50%).",
                "symptoms": "Early leaf spot: circular brown spots (2-10mm) with yellow halos. Late leaf spot: darker brown to black, smaller (1-6mm) spots, often with less prominent halos. Lesions on stems and pegs.",
                "conditions_favoring": "Warm (20-30°C) and humid conditions (>90% RH) or prolonged leaf wetness (8-12 hours).",
                "disease_cycle": "Fungi overwinter on crop residue, producing spores that infect new plants. Multiple cycles can occur within a season.",
                "impact": "Severe defoliation leading to poor pod development, reduced oil content, and lower yields.",
                "season": "Common from mid-season onwards (August-October in India).",
                "organic_options": ["Copper-based fungicides, Neem oil, rotation with non-host crops.", "Timely removal of crop residue."],
                "chemical_options": ["Fungicides like chlorothalonil, azoxystrobin, fluopyram. Timely application is critical."],
                "recommended_varieties": ["Some varieties show improved resistance like GG 20, TG 37A. Consult local agricultural universities."],
                "next_steps": [
                    "Implement a robust crop rotation program (2-3 years).",
                    "Apply fungicides preventatively, especially during high-risk weather.",
                    "Ensure adequate plant spacing for good air movement.",
                    "Sanitize fields by burying or removing infected crop residue."
                ],
                "external_links": {
                    "UF/IFAS Extension - Peanut Diseases": "https://sfyl.ifas.ufl.edu/media/sfylifasufledu/stlucie/docs/pdf/agricultural/Peanut_diseases_final.pdf",
                    "Researchgate - Groundnut Leaf Spots": "https://www.researchgate.net/publication/338908852_Management_of_Early_and_Late_Leaf_Spot_Diseases_of_Groundnut"
                }
            },
            "Peanut_healthy_leaf": {
                "cause": "N/A",
                "severity": "N/A",
                "symptoms": "Uniformly green, turgid leaves with no visible lesions, discoloration, or deformities. Vigorous plant growth.",
                "conditions_favoring": "Optimal soil moisture, balanced nutrients, full sunlight, and moderate temperatures.",
                "disease_cycle": "N/A",
                "impact": "Ensuring maximum productivity and high-quality kernels.",
                "season": "All growing seasons with optimal cultural practices.",
                "organic_options": ["Regular soil testing, organic amendments like compost, biological pest management."],
                "chemical_options": ["Balanced NPK fertilization, integrated pest management."],
                "recommended_varieties": ["All healthy varieties."],
                "next_steps": [
                    "Continue proper soil testing and nutrient management.",
                    "Maintain consistent irrigation to avoid water stress.",
                    "Monitor fields for early signs of disease or pests.",
                    "Ensure proper harvest practices to minimize post-harvest issues."
                ],
                "external_links": {
                    "ICAR - Groundnut Production": "https://www.icar.org.in/sites/default/files/Groundnut-Production-Technologies.pdf",
                    "National Peanut Board - Growing Peanuts": "https://nationalpeanutboard.org/peanut-facts/growing-peanuts/"
                }
            }
        }
    },
    "Tomato": {
        "model_path": "resnet50_tomato.pt",
        "labels": ["Tomato__Tomato_mosaic_virus", "Tomato___Bacterial_spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato__blight"],
        "num_classes": 4,
        "fc_type": "single_linear",
        "remedies": {
            "Tomato__Tomato_mosaic_virus": {
                "cause": "Highly contagious viral infection (ToMV) spread mechanically through sap, contaminated tools, hands, and even tobacco products. The virus is highly stable.",
                "severity": "Medium to High: Causes mottled, curled leaves and can significantly reduce fruit yield (up to 25%) and quality, making fruits un marketable.",
                "symptoms": "Mosaic-like patterns (light and dark green patches) on leaves, leaf curling and distortion, stunted plant growth, irregular fruit ripening (blotchy ripening).",
                "conditions_favoring": "No specific environmental conditions; spread by human activity and mechanical contact. Higher incidence in greenhouse settings.",
                "disease_cycle": "Virus particles transfer from infected to healthy plants via physical contact. Can survive in plant debris and seeds.",
                "impact": "Reduced yield, poor fruit quality, and potential spread to other solanaceous crops.",
                "season": "Any time, but often more noticeable in warmer growing periods.",
                "organic_options": ["Strict sanitation (disinfect tools, wash hands).", "Remove infected plants.", "Use virus-free seeds."],
                "chemical_options": ["No direct chemical cure for viruses. Focus on prevention and sanitation."],
                "recommended_varieties": ["Many resistant (Tm-2/Tm-2a gene) or tolerant varieties are available, e.g., 'Celebrity', 'Big Beef'. Check local seed catalogs."],
                "next_steps": [
                    "Remove and destroy any suspicious plants immediately.",
                    "Disinfect all tools and hands after working with plants.",
                    "Avoid using tobacco products near tomato plants.",
                    "Plant resistant varieties if possible."
                ],
                "external_links": {
                    "UC Davis IPM - Tomato Mosaic Virus": "https://www.ipm.ucdavis.edu/agriculture/tomato/tomato-mosaic-virus/",
                    "ICAR-IIVR - Tomato Production (India)": "https://iivr.icar.gov.in/tomato-production.html"
                }
            },
            "Tomato___Bacterial_spot": {
                "cause": "Caused by several species of *Xanthomonas* bacteria. Spread primarily by splashing water (rain, irrigation) from infected plant debris, soil, or seeds.",
                "severity": "Medium: Can cause significant defoliation and fruit spoilage, leading to reduced marketability and yield (up to 50%).",
                "symptoms": "Small, dark, water-soaked spots on leaves (1-3mm) that turn brown/black, often with a yellow halo. Raised, crusty, dark brown spots on fruits.",
                "conditions_favoring": "Warm (25-30°C) and humid conditions, prolonged leaf wetness, and heavy dews or rain.",
                "disease_cycle": "Bacteria survive in seeds, plant debris, or volunteer plants. Spread by rain, wind, and human activity. Enter through natural openings or wounds.",
                "impact": "Reduced yield, poor fruit quality, and increased susceptibility to other pathogens.",
                "season": "Warm, rainy season (Monsoon in India, July-October) and periods of high humidity.",
                "organic_options": ["Copper-based bactericides (e.g., Bordeaux mixture), bio-fungicides (e.g., *Bacillus amyloliquefaciens*).", "Ensure good air circulation, avoid overhead irrigation."],
                "chemical_options": ["Copper-based bactericides (e.g., copper hydroxide, copper oxychloride) are the primary chemical control. Resistance can occur."],
                "recommended_varieties": ["Some hybrid varieties offer partial resistance, but none are fully immune. Check local seed suppliers."],
                "next_steps": [
                    "Use certified disease-free seeds or transplants.",
                    "Apply copper-based sprays preventatively, especially during wet periods.",
                    "Ensure good plant spacing and pruning for air circulation.",
                    "Practice crop rotation (at least 2-3 years) with non-solanaceous crops."
                ],
                "external_links": {
                    "Cornell Extension - Bacterial Spot of Tomato": "https://www.vegetables.cornell.edu/pest-management/disease-factsheets/bacterial-spot-of-tomato/",
                    "ECHO Community - Managing Bacterial Spot": "https://www.echocommunity.org/resources/0e37456d-e43a-4467-8977-628d0979a405"
                }
            },
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
                "cause": "A viral disease (TYLCV) transmitted exclusively by the silverleaf whitefly (*Bemisia tabaci*). The virus is persistent within the whitefly vector.",
                "severity": "High: Leads to severe stunting, yellowing, and significantly reduced or no fruit production (near 100% loss in severe cases).",
                "symptoms": "Leaves curl upwards and inwards, become stunted and yellow (especially at the margins). Stunted plant growth, flowers may drop, few or no fruits develop.",
                "conditions_favoring": "Presence of whiteflies and warm temperatures (25-35°C). Spread is faster in dry, hot conditions that favor whitefly populations.",
                "disease_cycle": "Whiteflies acquire the virus from infected plants, then transmit it to healthy plants during feeding. The virus cannot be mechanically transmitted.",
                "impact": "Devastating yield losses, un marketable fruit, and challenges in controlling the vector.",
                "season": "Warm and dry seasons, when whitefly populations are high (e.g., summer and post-monsoon in India).",
                "organic_options": ["Control whiteflies using insecticidal soaps, neem oil, or horticultural oils. Use reflective mulches. Install sticky traps. Implement row covers."],
                "chemical_options": ["Apply systemic insecticides effective against whiteflies (e.g., neonicotinoids - use judiciously due to resistance concerns)."],
                "recommended_varieties": ["Many resistant (Ty-1/Ty-3 genes) varieties are available, e.g., 'Tycoon', 'Mountain Magic'. Check local seed suppliers for region-specific resistance."],
                "next_steps": [
                    "Plant resistant tomato varieties if available.",
                    "Implement whitefly control measures early and consistently.",
                    "Remove and destroy infected plants promptly to reduce virus source.",
                    "Consider growing in screened structures in high-pressure areas."
                ],
                "external_links": {
                    "UC Davis IPM - Tomato Yellow Leaf Curl Virus": "https://www.ipm.ucdavis.edu/agriculture/tomato/tomato-yellow-leaf-curl-virus/",
                    "Indian Institute of Horticultural Research (IIHR) - Tomato Viruses": "https://www.iihr.res.in/tomato-yellow-leaf-curl-virus-disease"
                }
            },
            "Tomato__blight": { # This can refer to early or late blight, here generalized
                "cause": "Early Blight (*Alternaria solani*): fungus surviving on debris, favored by warm, humid conditions. Late Blight (*Phytophthora infestans*): oomycete, highly aggressive, favored by cool, wet conditions.",
                "severity": "Medium to High: Can rapidly spread causing significant defoliation, fruit rot, and severe yield loss depending on the specific blight type.",
                "symptoms": "Early Blight: dark, concentric spots on older leaves. Late Blight: irregular, water-soaked lesions that rapidly turn brown/black, often with fuzzy growth on undersides.",
                "conditions_favoring": "Early Blight: 20-30°C, 6+ hours leaf wetness. Late Blight: 10-20°C, high humidity (>90%), prolonged leaf wetness (10+ hours).",
                "disease_cycle": "Fungi/oomycetes produce spores that are wind-borne or splash-dispersed, infecting new plants. Can survive in soil/debris/infected tubers (for late blight).",
                "impact": "Severe defoliation, fruit rot, and substantial reduction in yield and marketability.",
                "season": "Early Blight: Mid-season onwards. Late Blight: Cool, wet season (monsoon, winter).",
                "organic_options": ["Copper-based fungicides (e.g., Bordeaux mixture), biological fungicides.", "Pruning for air circulation.", "Crop rotation.", "Removing infected debris."],
                "chemical_options": ["Early Blight: Chlorothalonil, mancozeb. Late Blight: Systemic fungicides like propamocarb, dimethomorph, fluopicolide."],
                "recommended_varieties": ["Some resistant varieties are available for both early and late blight, e.g., 'Defiant PhR' (late blight), 'Mt. Spring' (early blight)."],
                "next_steps": [
                    "Apply appropriate fungicides preventatively during favorable conditions.",
                    "Ensure good plant spacing and pruning for air circulation.",
                    "Avoid overhead irrigation; water at the base of plants.",
                    "Remove and destroy all infected plant material thoroughly after harvest."
                ],
                "external_links": {
                    "Michigan State Extension - Managing Tomato Blights": "https://www.canr.msu.edu/news/managing_blight_on_tomatoes",
                    "SAU.ac.in - Tomato Diseases (India)": "https://www.sau.ac.in/tomato-production-and-diseases/"
                }
            }
        }
    },
    "Rice": {
        "model_path": "resnet50_rice.pt",
        "labels": ["Bacterialblight", "Brownspot", "Leafsmut"],
        "num_classes": 3,
        "fc_type": "custom_sequential_new_crop",
        "remedies": {
            "Bacterialblight": {
                "cause": "Caused by the bacterium *Xanthomonas oryzae pv. oryzae*. Spread by wind, rain, irrigation water, and direct contact. Enters through wounds or natural openings.",
                "severity": "Medium to High: Can cause significant yield reduction (10-30%) in susceptible varieties, particularly if infection occurs at seedling or tillering stages ('kresek').",
                "symptoms": "Water-soaked streaks on leaf margins that turn yellow then white/grey. Lesions may expand along the leaf, leading to wilting of entire seedlings ('kresek') or blighting of flag leaves.",
                "conditions_favoring": "Warm (25-30°C) and humid conditions (>70% RH), especially after heavy rainfall or strong winds. High nitrogen fertilization increases susceptibility.",
                "disease_cycle": "Bacteria survive in infected seeds, crop residue, or volunteer rice plants. Spread by water, wind, and tools. Penetrates through wounds or hydathodes.",
                "impact": "Reduced tillering, stunted growth, unfilled grains, and significant yield losses.",
                "season": "Typically prevalent during monsoon (wet season) in India (July-September).",
                "organic_options": ["Use resistant varieties. Seed treatment with beneficial microbes. Avoid excessive nitrogen. Improve field drainage."],
                "chemical_options": ["Seed treatment with antibiotics or copper compounds (use with caution). Foliar sprays with copper fungicides or streptomycin (where permitted)."],
                "recommended_varieties": ["Many resistant varieties like Pusa Basmati 1, IRBB series. Consult local agricultural universities/research centers."],
                "next_steps": [
                    "Plant resistant varieties where available.",
                    "Avoid excessive nitrogen fertilizer application.",
                    "Drain fields periodically to reduce bacterial spread.",
                    "Sanitize farm equipment and manage weed hosts."
                ],
                "external_links": {
                    "IRRI Rice Knowledge Bank - Bacterial Blight": "http://www.knowledgebank.irri.org/training/fact-sheets/disease-management/bacterial-blight",
                    "ICAR-NRRI - Rice Bacterial Blight (India)": "https://icar-nrri.in/diseases/bacterial_blight.html"
                }
            },
            "Brownspot": {
                "cause": "Caused by the fungus *Bipolaris oryzae* (*Cochliobolus miyabeanus*). Often indicative of nutritional deficiencies (especially potassium, calcium, magnesium, zinc) and poor soil health.",
                "severity": "Medium: Leads to reduced grain quality, lower milling recovery, and yield reduction (up to 45%) in severe cases, especially if panicles are infected.",
                "symptoms": "Oval or circular brown spots with a dark brown margin and light brown center on leaves, leaf sheaths, glumes, and grains. Spots may enlarge and coalesce.",
                "conditions_favoring": "High humidity (>85% RH), warm temperatures (25-30°C), and particularly nutrient-deficient soils. Flooding stress can exacerbate it.",
                "disease_cycle": "Fungus survives in infected seeds and crop debris. Spores are wind-borne and spread by splashing water. Multiple infection cycles can occur.",
                "impact": "Reduced grain quality and marketability due to blackened kernels. Slight reduction in overall yield.",
                "season": "Can occur throughout the growing season, but more pronounced during grain filling stage (September-October in India).",
                "organic_options": ["Improve soil fertility with organic matter. Balanced nutrient application (especially potassium).", "Use resistant varieties. Seed treatment with bio-control agents."],
                "chemical_options": ["Fungicides like carbendazim, propiconazole, or mancozeb. Apply during early disease onset or at panicle emergence."],
                "recommended_varieties": ["Some varieties show moderate resistance; 'Swarna', 'ADT 36' sometimes exhibit better tolerance."],
                "next_steps": [
                    "Ensure balanced fertilization, addressing any nutrient deficiencies.",
                    "Use certified disease-free seeds or treat seeds with fungicides.",
                    "Practice good field sanitation and remove infected plant debris.",
                    "Manage water levels to reduce plant stress."
                ],
                "external_links": {
                    "IRRI Rice Knowledge Bank - Brown Spot": "http://www.knowledgebank.irri.org/training/fact-sheets/disease-management/brown-spot",
                    "FAO - Brown Spot of Rice": "https://www.fao.org/plant-protection/assets/phytosanitary-capacity-development/plant-pests-diseases/rice-diseases/brown-spot/"
                }
            },
            "Leafsmut": {
                "cause": "Caused by the fungus *Tilletia horrida*. Infection occurs during flowering when spores get trapped within the florets and develop within the kernels.",
                "severity": "Low to Medium: Generally considered a minor disease, causing localized yield and quality losses (typically 1-5%). Primarily aesthetic damage to grains.",
                "symptoms": "Black, powdery spore masses that burst from infected kernels, causing a smutted appearance on panicles. Infected grains are transformed into a black, powdery mass; often only a few grains per panicle are affected.",
                "conditions_favoring": "High humidity and temperatures around 25-30°C during the booting and flowering stages. Light rain or dew facilitates spore germination.",
                "disease_cycle": "Spores overwinter on crop residue or in soil. Airborne spores infect flowers during pollination. The fungus grows inside the developing kernel.",
                "impact": "Reduced grain quality and marketability due to blackened kernels. Slight reduction in overall yield.",
                "season": "Symptoms become visible at milk or dough stage (October-November in India).",
                "organic_options": ["Use clean seeds. Rotate crops. Remove infected panicles before spores spread."],
                "chemical_options": ["Seed treatments with fungicides (e.g., trifloxystrobin, prochloraz) can reduce primary inoculum. Foliar fungicides at booting to heading stage if disease is severe."],
                "recommended_varieties": ["Generally, all commonly grown varieties are susceptible. Focus on prevention."],
                "next_steps": [
                    "Use certified, fungicide-treated seeds.",
                    "Remove and destroy infected panicles before they mature and release spores.",
                    "Practice crop rotation to reduce fungal inoculum in the soil.",
                    "Avoid excessive nitrogen fertilization, as it can worsen severity."
                ],
                "external_links": {
                    "IRRI Rice Knowledge Bank - Leaf Smut": "http://www.knowledgebank.irri.org/training/fact-sheets/disease-management/leaf-smut",
                    "University of Arkansas Rice Leaf Smut": "https://www.uaex.uada.edu/rice/rice-diseases/leaf-smut.aspx"
                }
            }
        }
    },
    "Cashew": { # UPDATED CROP
        "model_path": "resnet50_cashew.pt",
        "labels": ["Cashew anthracnose", "Cashew gummosis", "Cashew healthy", "Cashew leaf miner", "Cashew red rust"], # NEW LABELS
        "num_classes": 5, # UPDATED NUMBER OF CLASSES
        "fc_type": "single_linear", 
        "remedies": {
            "Cashew anthracnose": { # Updated label
                "cause": "Caused by the fungus *Colletotrichum gloeosporioides*. It affects young leaves, flowers, and fruits, leading to significant yield losses.",
                "severity": "High: Can cause extensive damage to tender shoots, inflorescences, and young nuts, leading to significant yield reduction.",
                "symptoms": "Dark brown to black spots on young leaves, flower panicles, and tender shoots. These spots can coalesce, leading to blight, drying, and shedding of affected parts. Pinkish spore masses may be visible in humid conditions.",
                "conditions_favoring": "High humidity (above 90%) and frequent rainfall, especially during flushing and flowering periods (warm and wet weather).",
                "disease_cycle": "Fungus overwinters on diseased plant parts. Spores are spread by wind and splashing rain, infecting new tissues. Multiple cycles occur during wet seasons.",
                "impact": "Reduced cashew nut yield and quality due to flower and fruit drop, and stunted growth of shoots.",
                "season": "Predominant during the monsoon and post-monsoon periods (June to November in India), coinciding with new flushes and flowering.",
                "organic_options": ["Pruning and destroying infected plant parts.", "Application of bio-fungicides like *Trichoderma harzianum* or *Pseudomonas fluorescens*.", "Neem oil spray for milder infections."],
                "chemical_options": ["Foliar sprays with systemic fungicides like Carbendazim, Hexaconazole, Propiconazole, or contact fungicides like Mancozeb, Bordeaux mixture. Regular prophylactic sprays are crucial."],
                "recommended_varieties": ["Varieties with some tolerance include VRI-3, Dhana, BPP-8. Consult local research institutes for region-specific resistant types."],
                "next_steps": [
                    "Regularly inspect trees, especially young flushes and flowers, for early symptoms.",
                    "Prune and destroy infected branches and fallen leaves to reduce inoculum.",
                    "Apply fungicides preventatively, especially during favorable weather conditions (new flush and flowering).",
                    "Ensure good air circulation within the canopy by proper tree spacing and pruning."
                ],
                "external_links": {
                    "ICAR-DCR - Diseases of Cashew": "https://dirc.icar.gov.in/diseases_cashew.html",
                    "ResearchGate - Anthracnose in Cashew": "https://www.researchgate.net/publication/322896577_Anthracnose_of_cashew_A_comprehensive_review"
                }
            },
            "Cashew gummosis": { # NEW REMEDY ADDED
                "cause": "Often caused by fungal pathogens (e.g., *Phytophthora* spp., *Botryosphaeria* spp.) or bacterial infections, frequently associated with wounds, stress, or waterlogging. It's a symptom rather than a single disease.",
                "severity": "Medium to High: Can lead to bark cracking, canker formation, branch dieback, and in severe cases, tree decline due to impaired nutrient and water transport.",
                "symptoms": "Exudation of gummy, amber-colored sap from cracks in the bark or lesions on the trunk and branches. Discoloration and necrosis of bark tissues. Leaves may yellow and drop prematurely.",
                "conditions_favoring": "High humidity, prolonged wet conditions, poor drainage, mechanical injuries, and root damage.",
                "disease_cycle": "Pathogens enter through wounds (pruning, mechanical, insect damage). Gummy exudation is a plant defense response to infection. Can spread through splashing water.",
                "impact": "Reduced tree vigor, canopy decline, and eventual tree death, leading to significant yield losses and shortened tree lifespan.",
                "season": "More prevalent during and after monsoon season (July-December in India) when soil moisture is high and humidity favors pathogen growth.",
                "organic_options": ["Improve drainage around trees.", "Avoid injuring bark during cultural operations.", "Prune and scrape off infected bark, then apply Bordeaux paste or bio-fungicides like *Trichoderma* spp.", "Ensure proper nutrition to enhance tree immunity."],
                "chemical_options": ["Drenching the soil with fungicides like Potassium Phosphonate or Metalaxyl-M for *Phytophthora*-related gummosis. Scraping infected bark and painting with fungicidal paste (e.g., Copper Oxychloride)."],
                "recommended_varieties": ["No truly resistant varieties, but healthy, vigorous trees are less susceptible. Good cultural practices are key."],
                "next_steps": [
                    "Inspect trunks and main branches regularly for gum exudation.",
                    "Improve soil drainage and avoid waterlogging.",
                    "Protect trees from mechanical injuries.",
                    "Promptly treat any wounds or signs of gummosis with appropriate fungicidal paste or drench."
                ],
                "external_links": {
                    "ICAR-DCR - Diseases of Cashew": "https://dirc.icar.gov.in/diseases_cashew.html", # General link for cashew diseases
                    "ResearchGate - Cashew Gummosis": "https://www.researchgate.net/publication/285587784_Gummosis_disease_in_cashew_caused_by_Phytophthora_species_and_its_management"
                }
            },
            "Cashew healthy": { # Updated label
                "cause": "N/A",
                "severity": "N/A",
                "symptoms": "Lush, vibrant green leaves without spots, lesions, or curling. Strong, vigorous growth of shoots and healthy flower and nut development. Absence of any visible disease signs.",
                "conditions_favoring": "Optimal sunlight, well-drained soil, balanced nutrient availability, adequate moisture, and proper cultural practices.",
                "disease_cycle": "N/A",
                "impact": "Achieving maximum potential yield of high-quality cashew nuts.",
                "season": "All growing seasons with optimal tree care and management.",
                "organic_options": ["Regular application of organic compost and manure.", "Use of bio-fertilizers to enhance nutrient uptake.", "Integrated pest management strategies focusing on natural predators."],
                "chemical_options": ["Balanced NPK fertilization based on soil analysis.", "Regular monitoring for pests and diseases, applying targeted chemical treatments only if necessary."],
                "recommended_varieties": ["All healthy, well-maintained cashew varieties."],
                "next_steps": [
                    "Continue balanced nutrition and appropriate irrigation based on tree needs and local climate.",
                    "Maintain good orchard hygiene by removing weeds and debris.",
                    "Conduct annual pruning to maintain tree shape, promote air circulation, and remove dead/diseased branches.",
                    "Regularly monitor for any early signs of pests or diseases to allow for timely intervention."
                ],
                "external_links": {
                    "CashewInfo - Cultivation Practices": "https://cashewinfo.com/cashew-cultivation-practices/",
                    "ICAR-DCR - Recommended Practices": "https://dirc.icar.gov.in/production_technology.html"
                }
            },
            "Cashew leaf miner": { # NEW REMEDY ADDED (assuming it's a pest, but common to include in disease detection apps)
                "cause": "Caused by the larvae of the cashew leaf miner moth (*Acrocercops syngramma*). The larvae tunnel within the leaf tissue, causing damage.",
                "severity": "Medium: Can cause defoliation of young leaves, especially new flushes, leading to reduced photosynthetic area and stunted growth. Severe infestations can impact yield.",
                "symptoms": "Serpentine or blotch-like tunnels (mines) visible on the leaf surface, created by feeding larvae. Affected leaves may curl, distort, dry up, and drop prematurely. Young shoots may be distorted.",
                "conditions_favoring": "Warm and humid conditions, especially during the flushing period when new tender leaves are available. Lack of natural enemies.",
                "disease_cycle": "Adult moths lay eggs on tender leaves. Larvae hatch and burrow into the leaf, feeding internally. Pupation occurs within the mine or on the leaf surface. Multiple generations per year.",
                "impact": "Reduced new shoot growth, defoliation of young leaves, and overall weakening of the tree, which can indirectly affect flowering and fruiting.",
                "season": "Active during the periods of new flush production, typically after monsoon or during the drier periods (August-March in India), but can vary by region.",
                "organic_options": ["Hand-picking and destroying infested leaves (for small scale).", "Application of Neem oil or other botanical insecticides.", "Encouraging natural predators like parasitic wasps.", "Use of sticky traps to monitor adult moths."],
                "chemical_options": ["Systemic insecticides like Imidacloprid or contact insecticides like Chlorpyrifos can be used, targeting young flushes and new infestations. Apply judiciously to protect beneficial insects."],
                "recommended_varieties": ["No specific resistant varieties; integrated pest management is key."],
                "next_steps": [
                    "Monitor new flushes closely for signs of mining activity.",
                    "Timely application of appropriate insecticides or organic treatments during infestation periods.",
                    "Prune affected leaves and young shoots to reduce pest population.",
                    "Maintain tree vigor to help tolerate pest pressure."
                ],
                "external_links": {
                    "ICAR-DCR - Pests of Cashew": "https://dirc.icar.gov.in/pests_cashew.html", # General link for cashew pests
                    "ResearchGate - Cashew Leaf Miner": "https://www.researchgate.net/publication/262791696_Management_of_cashew_leaf_miner_Acrocercops_syngramma_Meyr_Lepidoptera_Gracillariidae_in_cashew_Anacardium_occidentale_L"
                }
            },
            "Cashew red rust": { # Updated label
                "cause": "Caused by the parasitic green alga *Cephaleuros virescens*. It is often mistaken for a fungal disease but is an algal infection, favored by high humidity and poor air circulation.",
                "severity": "Low to Medium: Primarily affects leaves and young twigs, causing defoliation and weakening the tree. Can reduce photosynthetic efficiency and overall vigor but rarely causes tree death directly.",
                "symptoms": "Velvety, reddish-brown, circular patches (algal thalli) on the upper surface of leaves, often turning orange or rusty brown over time. Can also appear on young stems and nuts. Lesions may develop as a result of secondary fungal infections.",
                "conditions_favoring": "High humidity, prolonged leaf wetness, stagnant air, and shade. Poorly drained soils or dense canopies promote its development.",
                "disease_cycle": "Alga produces spores that are spread by wind and splashing rain. It penetrates the leaf cuticle and forms thalli on the surface. Can persist in perennial plant parts.",
                "impact": "Reduced photosynthetic area, premature leaf drop, and weakening of the tree, potentially affecting flowering and nut development in severe cases.",
                "season": "Predominant during the wet season (monsoon, July-September in India) when humidity is high.",
                "organic_options": ["Pruning to improve air circulation and sunlight penetration.", "Application of copper-based fungicides (e.g., Bordeaux mixture) as a cultural practice, though primarily algal.", "Neem oil can deter growth."],
                "chemical_options": ["Sprays of copper oxychloride or Bordeaux mixture are effective in controlling the algal growth. Repeat applications may be necessary during favorable conditions."],
                "recommended_varieties": ["No specific resistant varieties; cultural practices are key to management."],
                "next_steps": [
                    "Improve air circulation within the canopy through proper pruning and spacing.",
                    "Remove and destroy heavily infected leaves and twigs.",
                    "Apply copper fungicides during periods of high humidity, especially if symptoms are visible.",
                    "Avoid planting in overly dense plantations or poorly drained areas."
                ],
                "external_links": {
                    "ICAR-DCR - Red Rust of Cashew": "https://dirc.icar.gov.in/red_rust_cashew.html",
                    "Plant Protection - Algal Red Rust": "https://www.plantprotection.org/journal/index.php/jpps/article/view/178/142"
                }
            }
        }
    }
}
# --- END ENRICHED MODEL_INFO DICTIONARY ---


@st.cache_resource
def load_and_configure_model(path, num_classes_for_model, fc_layer_type):
    model = resnet50(weights=None)
    num_ftrs = model.fc.in_features

    if fc_layer_type == "single_linear":
        model.fc = nn.Linear(num_ftrs, num_classes_for_model)
    elif fc_layer_type == "custom_sequential_peanut":
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes_for_model)
        )
    elif fc_layer_type == "custom_sequential_new_crop":
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes_for_model)
        )
    else:
        raise ValueError(f"Unknown FC layer type: {fc_layer_type} for crop {crop}. Please check model_info configuration.")

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

current_model_path = model_info[crop]["model_path"]
labels = model_info[crop]["labels"]
current_num_classes = model_info[crop]["num_classes"]
fc_type = model_info[crop]["fc_type"]
static_remedies = model_info[crop]["remedies"]

model = load_and_configure_model(current_model_path, current_num_classes, fc_type)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Upload an image of the leaf", type=["jpg", "jpeg", "png"])
if not uploaded_file:
    uploaded_file = st.camera_input("Or take a picture")

prediction = None
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_idx = torch.argmax(output, 1).item()
        prediction = labels[predicted_idx]

    st.success(f"🧠 Predicted Disease: **{prediction}**")

    st.subheader("📋 Suggested Remedy (Expert Advice):")
    
    if prediction in static_remedies:
        remedy_data = static_remedies[prediction]
        st.markdown(f"### For {prediction}:")

        with st.expander("📊 Overview: Cause, Symptoms & Severity", expanded=True):
            if "cause" in remedy_data:
                st.markdown(f"**🔬 Cause:** {remedy_data['cause']}")
            if "symptoms" in remedy_data:
                st.markdown(f"**🔍 Key Symptoms:** {remedy_data['symptoms']}")
            if "severity" in remedy_data:
                severity_level = remedy_data['severity'].split(':')[0].strip().lower()
                color = "green"
                if "high" in severity_level:
                    color = "red"
                elif "medium" in severity_level:
                    color = "orange"
                st.markdown(f"**📈 Severity:** <span style='color:{color}'>{remedy_data['severity']}</span>", unsafe_allow_html=True)
            if "impact" in remedy_data:
                st.markdown(f"**📉 Impact on Yield:** {remedy_data['impact']}")
            if "season" in remedy_data:
                st.markdown(f"**🗓️ Typical Season:** {remedy_data['season']}")
            if "conditions_favoring" in remedy_data:
                st.markdown(f"**🌧️ Conditions Favoring:** {remedy_data['conditions_favoring']}")
            if "disease_cycle" in remedy_data:
                st.markdown(f"**🔄 Disease Cycle (Brief):** {remedy_data['disease_cycle']}")

        with st.expander("🛠️ Treatment & Mitigation"):
            if "mitigation" in remedy_data:
                st.markdown(f"**General Advice:** {remedy_data['mitigation']}")
            
            if "organic_options" in remedy_data and remedy_data["organic_options"]:
                st.markdown("**🌱 Organic/Biological Options:**")
                for opt in remedy_data["organic_options"]:
                    st.markdown(f"- {opt}")
            
            if "chemical_options" in remedy_data and remedy_data["chemical_options"]:
                st.markdown("**🧪 Chemical Options:**")
                for opt in remedy_data["chemical_options"]:
                    st.markdown(f"- {opt}")
            
            if "recommended_varieties" in remedy_data and remedy_data["recommended_varieties"]:
                st.markdown("**🌳 Recommended Resistant Varieties:**")
                for var in remedy_data["recommended_varieties"]:
                    st.markdown(f"- {var}")
            
            if isinstance(remedy_data, dict):
                if "Fertilizer" in remedy_data:
                    st.markdown(f"**🌱 Specific Fertilizer:** {remedy_data['Fertilizer']}")
                if 'Fertilizer_Calendar' in remedy_data:
                    st.markdown(f"**📅 Fertilizer Calendar Plan:** {remedy_data['Fertilizer_Calendar']}")

        if "next_steps" in remedy_data and remedy_data["next_steps"]:
            with st.expander("✅ Your Next Steps Checklist"):
                st.markdown("Mark these steps as you complete them:")
                for i, step in enumerate(remedy_data["next_steps"]):
                    st.checkbox(step, key=f"{prediction}_step_{i}")

        if "external_links" in remedy_data and remedy_data["external_links"]:
            with st.expander("🔗 Further Resources"):
                st.markdown("For more detailed information, visit these reputable sources:")
                for name, url in remedy_data["external_links"].items():
                    st.markdown(f"- [{name}]({url})")
    else:
        st.warning("No specific static remedy information available for this prediction.")


# --- START AI-POWERED ADVICE IN SIDEBAR ---
with st.sidebar:
    st.subheader("🤖 AI-Powered Advice (Google Gemini)")
    
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        try:
            GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
        except KeyError:
            st.warning("GEMINI_API_KEY not found in environment variables or Streamlit secrets.")
            GEMINI_API_KEY = None 

    # Function to clean markdown from text
    def clean_markdown(text):
        # Remove bold/italic markers (** and *)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text) # For **bold**
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # For *italic*
        text = re.sub(r'__([^_]+)__', r'\1', text)  # For __bold__
        text = re.sub(r'_([^_]+)_', r'\1', text)    # For _italic_

        # Remove bullet points from the beginning of lines (- or *)
        text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)

        # Remove bolding around specific keywords, but keep the keywords.
        # This is more precise than a general asterisk removal.
        text = text.replace('**Cause:**', 'Cause:')
        text = text.replace('**Treatment:**', 'Treatment:')
        text = text.replace('**Prevention:**', 'Prevention:')
        text = text.replace('**Impact on Yield:**', 'Impact on Yield:')
        text = text.replace('**Typical Season:**', 'Typical Season:')
        text = text.replace('**Conditions Favoring:**', 'Conditions Favoring:')
        text = text.replace('**Disease Cycle (Brief):**', 'Disease Cycle:')

        # Handle links: [link text](url) -> link text
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)

        # Corrected: Use re.sub for these replacements
        text = re.sub(r'### ', '', text) # Remove h3
        text = re.sub(r'## ', '', text)  # Remove h2
        text = re.sub(r'# ', '', text)   # Remove h1
        
        # Remove any remaining asterisks that might be part of scientific names (e.g., *Guignardia bidwellii*)
        # This is a bit more aggressive. If you want to keep "Guignardia bidwellii" as a single spoken phrase,
        # you might need a more sophisticated parser or to educate the TTS engine.
        # For simplicity, this will remove them.
        text = text.replace('*', '') # Use .replace() here as it's a literal string

        # Clean up extra spaces that might result from removals
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # Function to speak the text in a separate thread
    def speak_text(text_to_speak):
        cleaned_text = clean_markdown(text_to_speak)
        try:
            engine = pyttsx3.init()
            engine.say(cleaned_text)
            engine.runAndWait()
        except Exception as tts_e:
            st.warning(f"Text-to-speech error (server-side only): {tts_e}")

    # Callback function to handle the follow-up question submission
    def handle_follow_up_submission():
        user_follow_up_q = st.session_state.sidebar_follow_up_input_value
        if user_follow_up_q:
            st.session_state.sidebar_messages.append({"role": "user", "parts": [{"text": user_follow_up_q}]})
            
            # Prepare chat history for Gemini (excluding the current user question, as it's handled below)
            gemini_chat_history_for_new_turn = []
            for msg in st.session_state.sidebar_messages[:-1]: # Exclude the most recent user message initially
                if msg["role"] in ["user", "model"]:
                    gemini_chat_history_for_new_turn.append({"role": msg["role"], "parts": msg["parts"]})

            concise_follow_up_prompt = f"""
            You are an experienced agricultural expert providing highly concise and practical advice to farmers.
            Please answer the following question about {crop} plants.
            Keep the entire response very short and to the point (under 80 words). Avoid any introductory or concluding sentences.

            User's question: {user_follow_up_q}
            """
            
            try:
                chat = model_gemini.start_chat(history=gemini_chat_history_for_new_turn)
                follow_up_resp = chat.send_message(concise_follow_up_prompt)
                ai_response_text = follow_up_resp.text
                
                # Append the AI's response to the session state messages
                st.session_state.sidebar_messages.append({"role": "model", "parts": [{"text": ai_response_text}]})
                
                # Clear the input box value in session state after processing
                st.session_state.sidebar_follow_up_input_value = "" # Clears the widget on next rerun
            except Exception as e:
                st.error(f"❌ An error occurred with Google Gemini: {e}")
                st.session_state.sidebar_messages.append({"role": "model", "parts": [{"text": "Sorry, I couldn't generate a response at this time."}]})


    if GEMINI_API_KEY: 
        genai.configure(api_key=GEMINI_API_KEY)

        model_gemini = genai.GenerativeModel('models/gemma-3-4b-it') 

        # Initialize chat history and a flag to control initial AI response generation
        # Reset history and flag if crop changes or no prediction is active
        if "sidebar_messages" not in st.session_state or st.session_state.get("last_prediction_for_ai", None) != prediction:
            st.session_state.sidebar_messages = []
            st.session_state.ai_advice_initial_generated = False
            st.session_state.last_prediction_for_ai = prediction # Store current prediction to detect changes
            st.session_state.sidebar_follow_up_input_value = "" # Initialize input field value for new session

        # Only show toggle if a prediction has been made in the main content
        if prediction:
            if st.toggle("💬 Get AI-generated advice", key="ai_advice_toggle"):
                if not st.session_state.ai_advice_initial_generated:
                    with st.spinner("Contacting AI expert..."):
                        prompt_text = f"""
                        You are an experienced agricultural expert providing highly concise and practical advice to farmers.
                        My {crop} plant has {prediction}.

                        Provide actionable advice, formatted clearly with bullet points.
                        Keep the entire response very short and to the point (under 80 words). Avoid any introductory or concluding sentences.

                        Focus on:
                        - **Cause:** (1 sentence max)
                        - **Treatment:** (1-2 very short points)
                        - **Prevention:** (1-2 very short points)
                        """
                        try:
                            chat = model_gemini.start_chat(history=[])
                            response = chat.send_message(prompt_text) 
                            reply = response.text
                            
                            st.session_state.sidebar_messages.append({"role": "model", "parts": [{"text": reply}]})
                            st.session_state.ai_advice_initial_generated = True

                        except Exception as e:
                            st.error(f"❌ An error occurred with Google Gemini: {e}")
                            st.session_state.ai_advice_initial_generated = False
                
                # Display current chat history in the sidebar for context
                for message_idx, message in enumerate(st.session_state.sidebar_messages):
                    if message["role"] == "user":
                        st.markdown(f"**You:** {message['parts'][0]['text']}")
                    elif message["role"] == "model":
                        st.info(message['parts'][0]['text'])
                        # Add a "Listen" button for each AI response
                        if st.button(f"🔊 Listen to AI Advice {message_idx + 1}", key=f"listen_ai_response_{message_idx}"):
                            threading.Thread(target=speak_text, args=(message['parts'][0]['text'],)).start()

                st.subheader("💬 Ask a follow-up question")
                # Use a unique key for the input and its current value from session state
                st.text_input(
                    "Type your question here:",
                    key="sidebar_follow_up_input_value", # This key stores the actual value
                    on_change=handle_follow_up_submission, # Call the function when input changes (e.g., Enter is pressed)
                    placeholder="E.g., How often should I apply the fungicide?"
                )

            else:
                st.info("Toggle 'Get AI-generated advice' to enable AI features.")
                st.session_state.ai_advice_initial_generated = False
                st.session_state.sidebar_messages = []
                st.session_state.last_prediction_for_ai = None
                st.session_state.sidebar_follow_up_input_value = "" # Ensure cleared on toggle off

        else:
            st.info("Upload an image and get a prediction first to enable AI advice in the sidebar.")
            st.session_state.ai_advice_initial_generated = False
            st.session_state.sidebar_messages = []
            st.session_state.last_prediction_for_ai = None
            st.session_state.sidebar_follow_up_input_value = "" # Ensure cleared on no prediction

    else:
        st.info("Please set your Google Gemini API key as an environment variable (GEMINI_API_KEY) or in Streamlit secrets.")