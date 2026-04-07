import os
import csv
from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter

#==============================================
# 从materialsproject拉取素材
#==============================================

API_KEY = "rdq9JwSE1rePyRtCKlqS6ZQgGWcYoz9U"
SAVE_DIR = "../data/catalysis"

search_tasks = []

# 基础过渡金属氧化物
base_metals = ["Fe", "Mn", "Co", "Ni", "Cu", "Ti"]
for metal in base_metals:
    search_tasks.append({
        "name": f"{metal}-O",
        "elements": [metal, "O"],
        "num_elements": (2, 3),      # 二元或三元
        "is_stable": True
    })

# 贵金属氧化物
noble_metals = ["Pt", "Pd", "Rh", "Ru"]
for metal in noble_metals:
    search_tasks.append({
        "name": f"{metal}-O",
        "elements": [metal, "O"],
        "num_elements": (2, 3),
        "is_stable": True
    })

# 钙钛矿典型体系
perovskite_systems = [
    "La-Mn-O", "La-Co-O", "Sr-Ti-O", "Ba-Ti-O",
    "La-Fe-O", "La-Ni-O", "Ca-Ti-O", "Sr-Fe-O"
]
for chemsys in perovskite_systems:
    search_tasks.append({
        "name": f"perovskite_{chemsys}",
        "chemsys": chemsys,
        "is_stable": True
    })

# 尖晶石典型体系
spinel_systems = [
    "Co-Fe-O", "Ni-Fe-O", "Mn-Co-O", "Cu-Fe-O",
    "Zn-Fe-O", "Mg-Fe-O", "Ni-Mn-O"
]
for chemsys in spinel_systems:
    search_tasks.append({
        "name": f"spinel_{chemsys}",
        "chemsys": chemsys,
        "is_stable": True
    })

# 创建保存目录
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

all_docs = []
with MPRester(API_KEY) as mpr:
    for task in search_tasks:
        print(f"正在抓取: {task['name']} ...")
        params = {"is_stable": task["is_stable"]}
        if "elements" in task:
            params["elements"] = task["elements"]
        if "num_elements" in task:
            params["num_elements"] = task["num_elements"]
        if "chemsys" in task:
            params["chemsys"] = task["chemsys"]

        try:
            docs = mpr.materials.summary.search(
                **params,
                fields=["material_id", "structure", "formation_energy_per_atom", "band_gap"]
            )
            print(f"  找到 {len(docs)} 个材料")
            all_docs.extend(docs)
        except Exception as e:
            print(f"  任务失败: {e}")
            continue

# 按 material_id 去重
unique_docs = {doc.material_id: doc for doc in all_docs}.values()
print(f"\n总计找到 {len(unique_docs)} 个唯一材料，开始保存...")

# 保存 CIF 和 id_prop.csv
id_prop_data = []
for doc in unique_docs:
    mp_id = str(doc.material_id)
    cw = CifWriter(doc.structure)
    cw.write_file(os.path.join(SAVE_DIR, f"{mp_id}.cif"))
    id_prop_data.append([mp_id, doc.formation_energy_per_atom])

with open(os.path.join(SAVE_DIR, "id_prop.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["material_id", "formation_energy_per_atom"])
    writer.writerows(id_prop_data)

print(f"任务完成！共保存 {len(id_prop_data)} 组数据至: {SAVE_DIR}")