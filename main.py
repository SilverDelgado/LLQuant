from data import get_structured_data, get_unstructured_data,prepare_full_payload
# Importa la nueva función si la pusiste en data.py, o defínela aquí
# from data import prepare_full_payload 
import json

# --- USO ---
structured = get_structured_data("cmt_btcusdt", verbose=False)
unstructured = get_unstructured_data("cmt_btcusdt", verbose=False)

# Usamos la nueva función de fusión inteligente
final_llm_input = prepare_full_payload(structured, unstructured)

print(json.dumps(final_llm_input, indent=2))