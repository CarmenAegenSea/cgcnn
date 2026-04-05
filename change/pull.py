import os
import csv
from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter

API_KEY = "rdq9JwSE1rePyRtCKlqS6ZQgGWcYoz9U"
SAVE_DIR = "../data/catalysis"
TRANSITION_METALS = ["Fe", "Mn", "Co", "Ni", "Cu", "Ti"]

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

with MPRester(API_KEY) as mpr:
    all_docs = []
    for metal in TRANSITION_METALS:
        print(f"正在抓取 {metal}-O 体系...")
        docs = mpr.materials.summary.search(
            elements=[metal, "O"],
            is_stable=True,
            num_elements=(2, 3),
            fields=["material_id", "structure", "formation_energy_per_atom", "band_gap"]
        )
        all_docs.extend(docs)

    unique_docs = {doc.material_id: doc for doc in all_docs}.values()
    print(f"总计找到 {len(unique_docs)} 个唯一材料，准备保存...")

    id_prop_data = []
    for i, doc in enumerate(unique_docs):
        mp_id = str(doc.material_id)
        cw = CifWriter(doc.structure)

        cw.write_file(os.path.join(SAVE_DIR, f"{mp_id}.cif"))

        id_prop_data.append([mp_id, doc.formation_energy_per_atom])

    with open(os.path.join(SAVE_DIR, "id_prop.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(id_prop_data)

print(f"任务完成！共保存 {len(id_prop_data)} 组数据至: {SAVE_DIR}")