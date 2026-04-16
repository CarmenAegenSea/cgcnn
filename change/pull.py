"""
Materials Project 光催化剂数据拉取（竞赛实用版）

设计目标：快速获取一定数量（如 500-1000）的光催化剂候选材料，
        用于 CGCNN 模型训练与统计建模竞赛展示。
筛选条件：
    - 带隙：1.0 ~ 3.2 eV
    - 形成能：≤ 0.1 eV/atom
    - 稳定性：energy_above_hull ≤ 0.2 eV/atom
    - 仅排除放射性元素，其余元素全部保留
    - 不进行带边位置筛选
"""

import os
import json
import time
import csv
from typing import List, Optional, Dict, Any, Set
from pathlib import Path

from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter


# ==================== 配置区 ====================
API_KEY = "rdq9JwSE1rePyRtCKlqS6ZQgGWcYoz9U"          # 替换为你的 API 密钥
OUTPUT_DIR = "../data/candidates"
MAX_MATERIALS = 800                    # 目标候选数量（可根据需要调整）
REQUEST_DELAY = 0.1

# 核心筛选参数
BAND_GAP_MIN = 1.2                     # eV
BAND_GAP_MAX = 3.0                     # eV
FORMATION_ENERGY_MAX = 0.1             # eV/atom，允许轻微亚稳态
MAX_ENERGY_ABOVE_HULL = 0.2            # eV/atom，容忍轻微亚稳态

# 仅排除放射性元素（保证数据合法性）
EXCLUDED_ELEMENTS: Set[str] = {
    "Tc", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu"
}

# 输出控制
DOWNLOAD_CIF = True
SAVE_CSV = True
SAVE_JSON = True
# =================================================


def get_api_key() -> Optional[str]:
    """获取 API 密钥，优先使用脚本中设置的，其次从环境变量读取"""
    if API_KEY and API_KEY != "your_api_key_here":
        return API_KEY
    return os.environ.get("MP_API_KEY")


def build_query() -> Dict[str, Any]:
    """构建 MPRester 查询条件"""
    query = {
         "is_stable": True,
    }
    query["band_gap"] = (BAND_GAP_MIN, BAND_GAP_MAX)
    query["formation_energy_per_atom"] = (None, FORMATION_ENERGY_MAX)
    return query


def element_blacklist_filter(elements: List[str]) -> bool:
    """检查是否包含被排除的元素（放射性元素）"""
    if not elements:
        return False
    return any(elem in EXCLUDED_ELEMENTS for elem in elements)


class FilterStats:
    """筛选统计计数器"""
    def __init__(self):
        self.total = 0
        self.excluded_elements = 0
        self.unstable = 0
        self.passed = 0

    def print_summary(self):
        print("\n" + "=" * 60)
        print("筛选统计")
        print("=" * 60)
        print(f"检查材料总数: {self.total}")
        print(f"  - 包含排除元素: {self.excluded_elements}")
        print(f"  - 亚稳态剔除: {self.unstable}")
        print(f"最终通过数量: {self.passed}")
        print("=" * 60)


def fetch_candidates(mpr: MPRester, max_materials: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    从 Materials Project 拉取并筛选候选材料
    """
    print("=" * 60)
    print("光催化剂候选数据拉取（竞赛实用版）")
    print(f"带隙范围: {BAND_GAP_MIN} - {BAND_GAP_MAX} eV")
    print(f"形成能上限: {FORMATION_ENERGY_MAX} eV/atom")
    print(f"稳定性要求: energy_above_hull ≤ {MAX_ENERGY_ABOVE_HULL} eV/atom")
    print(f"排除元素: {sorted(EXCLUDED_ELEMENTS)}")
    print("=" * 60)

    query = build_query()
    print(f"API 基础查询: {query}")

    # 需要从 API 获取的字段
    fields = [
        "material_id", "formula_pretty", "band_gap", "formation_energy_per_atom",
        "energy_above_hull", "volume", "density", "nsites", "elements",
        "cbm", "vbm", "is_stable", "symmetry", "structure",
    ]

    candidates = []
    stats = FilterStats()

    try:
        # 使用新版 MPRester 的分页查询
        docs = mpr.materials.summary.search(
            **query,
            fields=fields,
            chunk_size=500,
        )

        for doc in docs:
            stats.total += 1
            if stats.total % 500 == 0:
                print(f"  已检查 {stats.total} 个材料，当前候选: {stats.passed}")

            # 提取元素列表
            elem_list = []
            if hasattr(doc, 'elements') and doc.elements:
                for e in doc.elements:
                    sym = e.symbol if hasattr(e, 'symbol') else str(e)
                    elem_list.append(sym)

            # 元素黑名单过滤
            if element_blacklist_filter(elem_list):
                stats.excluded_elements += 1
                continue

            # 稳定性过滤
            e_above_hull = getattr(doc, 'energy_above_hull', None)
            if e_above_hull is not None and e_above_hull > MAX_ENERGY_ABOVE_HULL:
                stats.unstable += 1
                continue

            # 提取晶体学信息
            crystal_system_str = None
            spacegroup_str = None
            if hasattr(doc, 'symmetry') and doc.symmetry:
                cs = getattr(doc.symmetry, 'crystal_system', None)
                if cs is not None:
                    crystal_system_str = cs.value if hasattr(cs, 'value') else str(cs)
                sg = getattr(doc.symmetry, 'symbol', None)
                if sg is not None:
                    spacegroup_str = str(sg)

            data = {
                "material_id": str(doc.material_id),
                "formula": getattr(doc, 'formula_pretty', None),
                "band_gap": getattr(doc, 'band_gap', None),
                "formation_energy_per_atom": getattr(doc, 'formation_energy_per_atom', None),
                "energy_above_hull": e_above_hull,
                "volume": getattr(doc, 'volume', None),
                "density": getattr(doc, 'density', None),
                "nsites": getattr(doc, 'nsites', None),
                "elements": elem_list,
                "cbm": getattr(doc, 'cbm', None),
                "vbm": getattr(doc, 'vbm', None),
                "crystal_system": crystal_system_str,
                "spacegroup": spacegroup_str,
                "is_stable": getattr(doc, 'is_stable', None),
                "structure": doc.structure if hasattr(doc, 'structure') else None,
            }

            candidates.append(data)
            stats.passed += 1

            if max_materials and stats.passed >= max_materials:
                break

            time.sleep(REQUEST_DELAY)

    except Exception as e:
        print(f"查询出错: {e}")
        import traceback
        traceback.print_exc()

    stats.print_summary()
    return candidates


def save_candidates(candidates: List[Dict[str, Any]], output_dir: Path):
    """保存候选材料数据及 CIF 文件"""
    if not candidates:
        print("没有候选材料，跳过保存。")
        return

    cif_dir = output_dir / "cif"
    cif_dir.mkdir(parents=True, exist_ok=True)

    light_data = []
    for item in candidates:
        # 去除无法序列化的 structure 对象
        light_item = {k: v for k, v in item.items() if k != "structure"}
        light_data.append(light_item)

        if DOWNLOAD_CIF and item.get("structure"):
            cif_path = cif_dir / f"{item['material_id']}.cif"
            try:
                CifWriter(item["structure"]).write_file(str(cif_path))
                light_item["cif_path"] = str(cif_path)
            except Exception as e:
                print(f"  保存 CIF 失败 {item['material_id']}: {e}")
                light_item["cif_path"] = None
        else:
            light_item["cif_path"] = None

    if SAVE_CSV:
        csv_path = output_dir / "candidates.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=light_data[0].keys())
            writer.writeheader()
            writer.writerows(light_data)
        print(f"CSV 已保存至: {csv_path}")

    if SAVE_JSON:
        json_path = output_dir / "candidates.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(light_data, f, indent=2, ensure_ascii=False)
        print(f"JSON 已保存至: {json_path}")

    print(f"CIF 文件总数: {len(list(cif_dir.glob('*.cif')))}")


def main():
    print("=" * 60)
    print("Materials Project 光催化剂数据拉取")
    print("=" * 60)

    api_key = get_api_key()
    if not api_key:
        print("错误：未找到 API 密钥！")
        print("请在脚本中设置 API_KEY 变量或设置环境变量 MP_API_KEY")
        return

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"数据保存目录: {output_dir}")

    try:
        with MPRester(api_key) as mpr:
            candidates = fetch_candidates(mpr, max_materials=MAX_MATERIALS)

            if candidates:
                save_candidates(candidates, output_dir)

                print("\n最终候选材料属性概要：")
                band_gaps = [c["band_gap"] for c in candidates if c["band_gap"] is not None]
                if band_gaps:
                    print(f"  带隙范围: {min(band_gaps):.2f} - {max(band_gaps):.2f} eV")
                form_es = [c["formation_energy_per_atom"] for c in candidates if c["formation_energy_per_atom"] is not None]
                if form_es:
                    print(f"  形成能范围: {min(form_es):.3f} - {max(form_es):.3f} eV/atom")
            else:
                print("未找到任何候选材料。")

    except Exception as e:
        print(f"执行错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()