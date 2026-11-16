# -*- coding: utf-8 -*-
"""NER 后处理函数模块。

从 main_show.py 中 MultiRegionApp 的静态方法提取，独立为模块以供 Flask 服务调用。
所有函数逻辑与原代码完全一致，仅将 MultiRegionApp.xxx 调用改为直接函数调用。
"""

import re

# ============== 常量 ==============
VALID_LABELS = {
    "姓名", "性别", "出生年月", "民族", "籍贯", "出生地",
    "政治面貌", "入党时间", "参加工作时间", "工作单位", "职务", "现职时间",
    "身份证号", "全日制教育学历", "全日制教育学位", "全日制教育毕业院校系", "全日制教育专业",
    "在职教育学历", "在职教育学位", "在职教育毕业院校系", "在职教育专业"
}

VALID_POLITICAL_STATUS = {
    "党员", "中共党员", "预备党员", "共青团员", "团员", "群众", "民主党派", "无党派"
}


# ============== 基础工具函数 ==============

def normalize_word(word: str) -> str:
    return word.replace("##", "").replace(" ", "")


def clean_entity_text(word: str) -> str:
    return re.sub(
        r'^[\s#;；,，。:：\|""\(\)（）\[\]【】<>《》]+|[\s#;；,，。:：\|""\(\)（）\[\]【】<>《》]+$',
        "", word,
    )


def is_symbol_only(word: str) -> bool:
    return bool(word) and re.fullmatch(r"[\W_#]+", word, flags=re.UNICODE) is not None


def is_year_month_like(word: str) -> bool:
    return re.fullmatch(
        r"(?:\d{4}|[〇零一二三四五六七八九十]{4})年(?:\d{1,2}|[〇零一二三四五六七八九十]{1,3})月", word
    ) is not None


def is_full_date_like(word: str) -> bool:
    return re.fullmatch(
        r"(?:\d{4}|[〇零一二三四五六七八九十]{4})年(?:\d{1,2}|[〇零一二三四五六七八九十]{1,3})月"
        r"(?:\d{1,2}|[〇零一二三四五六七八九十]{1,3})日",
        word,
    ) is not None


def is_year_like(word: str) -> bool:
    return re.fullmatch(r"(?:\d{4}|[〇零一二三四五六七八九十]{4})", word) is not None


def is_cn_id_number(word: str) -> bool:
    w = word.replace(" ", "")
    return (
        re.fullmatch(r"[1-9]\d{16}[0-9Xx]", w) is not None
        or re.fullmatch(r"[1-9]\d{14}", w) is not None
    )


def expand_year_to_year_month(word: str, text: str, start: int, end: int):
    if not (isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(text)):
        return word, end
    if not is_year_like(word):
        return word, end
    if end >= len(text) or text[end] != "年":
        return word, end
    max_right = min(len(text), end + 8)
    seg = text[start:max_right]
    m = re.match(
        r"^(?:\d{4}|[〇零一二三四五六七八九十]{4})年(?:\d{1,2}|[〇零一二三四五六七八九十]{1,3})月", seg
    )
    if m:
        expanded = m.group(0)
        return expanded, start + len(expanded)
    return word, end


def injected_score(items: list, start: int, end: int) -> float:
    nearby = []
    for it in items:
        s = it.get("start")
        e = it.get("end")
        sc = it.get("score")
        if not (isinstance(s, int) and isinstance(e, int)):
            continue
        if not isinstance(sc, (int, float)):
            continue
        dist = min(abs(start - s), abs(end - e))
        if dist <= 24:
            nearby.append(float(sc))
    if nearby:
        base = max(nearby)
        return round(min(0.999999, max(0.900001, base - 0.000001)), 12)
    return 0.999123456789


# ============== 过滤与修复函数 ==============

def should_drop_entity(label: str, word: str, text: str, start: int, end: int) -> bool:
    if not word:
        return True
    if is_symbol_only(word):
        return True
    if label not in VALID_LABELS:
        return True
    if label == "身份证号":
        clean_word = re.sub(r'[\s\-_]', '', word)
        is_valid_18 = re.fullmatch(r'[1-9]\d{16}[\dXx]', clean_word) is not None
        is_valid_15 = re.fullmatch(r'[1-9]\d{14}', clean_word) is not None
        if not (is_valid_18 or is_valid_15):
            return True
    if label == "政治面貌":
        if word not in VALID_POLITICAL_STATUS:
            return True
    if len(word) == 1 and label not in {"性别", "民族"}:
        return True
    noise_words = {"无", "不详", "未知", "待填", "市", "员", "技术"}
    if word in noise_words:
        return True
    if label == "全日制教育专业" and word in {"技术", "职务", "工资", "情况"}:
        return True
    if label == "入党时间":
        if not re.match(r"\d{4}[./年]\d{1,2}([./月]\d{1,2}日?)?", word):
            return True
        if any(k in word for k in ["会议", "全会", "召开", "期间", "十一届", "三中全会"]):
            return True
    if label in ["全日制教育毕业院校系", "在职教育毕业院校系"]:
        political_orgs = ["中国共产主义青年团", "共青团", "共产党", "党支部", "党委", "委员会"]
        if word in political_orgs or any(org in word for org in political_orgs):
            return True
    if label == "籍贯" and any(k in word for k in ["路", "号", "小区", "街道", "村", "胡同"]):
        return True
    if label in ["全日制教育学历", "在职教育学历"]:
        if word == "大学" or word == "大":
            return True
    return False


def fix_incomplete_dates(items: list, text: str) -> list:
    out = []
    date_labels = {"参加工作时间", "出生年月", "入党时间", "现职时间"}
    for item in items:
        label = str(item.get("entity_group", ""))
        word = str(item.get("word", ""))
        start = item.get("start")
        if label in date_labels and word.startswith("年") and isinstance(start, int):
            prev_text = text[max(0, start - 6):start]
            year_match = re.search(r'(\d{4})', prev_text)
            if year_match:
                year = year_match.group(1)
                full_word = year + word
                new_start = start - len(year)
                item["word"] = full_word
                item["start"] = new_start
        out.append(item)
    return out


def fix_adjacent_split_entities(items: list, text: str) -> list:
    if not items:
        return items
    fixed = []
    i = 0
    while i < len(items):
        cur = items[i]
        if i + 1 < len(items):
            nxt = items[i + 1]
            cur_start = cur.get("start")
            cur_end = cur.get("end")
            nxt_start = nxt.get("start")
            nxt_end = nxt.get("end")
            cur_word = str(cur.get("word", ""))
            nxt_word = str(nxt.get("word", ""))
            cur_label = str(cur.get("entity_group", ""))
            nxt_label = str(nxt.get("entity_group", ""))
            if (isinstance(cur_start, int) and isinstance(cur_end, int) and
                isinstance(nxt_start, int) and isinstance(nxt_end, int) and
                cur_end == nxt_start):
                # 合并公司后缀
                if cur_word.endswith("有限") and nxt_word == "公司":
                    merged = {
                        "entity_group": cur_label if cur_label in VALID_LABELS else nxt_label,
                        "score": max(float(cur.get("score", 0.0)), float(nxt.get("score", 0.0))),
                        "word": cur_word + nxt_word, "start": cur_start, "end": nxt_end,
                    }
                    fixed.append(merged); i += 2; continue
                if cur_word.endswith("建筑") and nxt_word.startswith("有限"):
                    merged = {
                        "entity_group": cur_label if cur_label in VALID_LABELS else nxt_label,
                        "score": max(float(cur.get("score", 0.0)), float(nxt.get("score", 0.0))),
                        "word": cur_word + nxt_word, "start": cur_start, "end": nxt_end,
                    }
                    fixed.append(merged); i += 2; continue
                if cur_word == "研" and nxt_word == "究生":
                    merged = {
                        "entity_group": "全日制教育学历",
                        "score": max(float(cur.get("score", 0.0)), float(nxt.get("score", 0.0))),
                        "word": "研究生", "start": cur_start, "end": nxt_end,
                    }
                    fixed.append(merged); i += 2; continue
                if cur_word == "大学" and nxt_word == "本科":
                    if cur_label in ["全日制教育学历", "在职教育学历"] and nxt_label in ["全日制教育学历", "在职教育学历"]:
                        merged = {
                            "entity_group": nxt_label, "score": nxt.get("score", 0.0),
                            "word": "本科", "start": nxt_start, "end": nxt_end,
                        }
                        fixed.append(merged); i += 2; continue
                if cur_word == "大学" and nxt_word == "专科":
                    if cur_label in ["全日制教育学历", "在职教育学历"] and nxt_label in ["全日制教育学历", "在职教育学历"]:
                        merged = {
                            "entity_group": nxt_label, "score": nxt.get("score", 0.0),
                            "word": "专科", "start": nxt_start, "end": nxt_end,
                        }
                        fixed.append(merged); i += 2; continue
                if cur_label == "在职教育毕业院校系" and nxt_label == "在职教育毕业院校系":
                    merged = {
                        "entity_group": "在职教育毕业院校系",
                        "score": max(float(cur.get("score", 0.0)), float(nxt.get("score", 0.0))),
                        "word": cur_word + nxt_word, "start": cur_start, "end": nxt_end,
                    }
                    fixed.append(merged); i += 2; continue
                if cur_label == "全日制教育毕业院校系" and nxt_label == "全日制教育毕业院校系":
                    merged = {
                        "entity_group": "全日制教育毕业院校系",
                        "score": max(float(cur.get("score", 0.0)), float(nxt.get("score", 0.0))),
                        "word": cur_word + nxt_word, "start": cur_start, "end": nxt_end,
                    }
                    fixed.append(merged); i += 2; continue
                # 日期合并
                combined_date = cur_word + nxt_word
                date_labels = {"出生年月", "参加工作时间", "现职时间", "入党时间"}
                if is_year_month_like(combined_date) and {cur_label, nxt_label}.issubset(date_labels):
                    merged_label = cur_label if cur_label != "日期" else (nxt_label if nxt_label != "日期" else "出生年月")
                    merged = {
                        "entity_group": merged_label,
                        "score": max(float(cur.get("score", 0.0)), float(nxt.get("score", 0.0))),
                        "word": combined_date, "start": cur_start, "end": nxt_end,
                    }
                    fixed.append(merged); i += 2; continue
                if is_full_date_like(combined_date) and {cur_label, nxt_label}.issubset(date_labels):
                    merged_label = cur_label if cur_label != "日期" else (nxt_label if nxt_label != "日期" else "出生年月")
                    merged = {
                        "entity_group": merged_label,
                        "score": max(float(cur.get("score", 0.0)), float(nxt.get("score", 0.0))),
                        "word": combined_date, "start": cur_start, "end": nxt_end,
                    }
                    fixed.append(merged); i += 2; continue
                if not (len(cur_word) == 1 and len(nxt_word) == 1):
                    fixed.append(cur); i += 1; continue
        fixed.append(cur)
        i += 1
    return fixed


def context_relabel(items: list, text: str) -> list:
    if not items:
        return items
    out = [dict(x) for x in items]
    school_keywords = ["大学", "学院", "学校", "研究院", "系", "技术学院", "师范大学", "理工大学", "工业大学", "科技大学"]
    fulltime_keywords = ["全日制", "全日制教育", "全日制学历", "毕业院校", "学历学位", "全日制毕业"]
    parttime_keywords = ["在职", "在职教育", "在职学历", "继续教育", "成人教育"]
    family_keywords = ["配偶", "父亲", "母亲", "妹妹", "哥哥", "弟弟", "姐姐", "子女", "关系"]
    work_history_keywords = ["参加工作", "工作经历", "简历", "学习和工作", "至今", "前后简历"]

    family_region_start = -1
    for kw in family_keywords:
        pos = text.find(kw)
        if pos != -1:
            family_region_start = pos
            break

    for item in out:
        label = str(item.get("entity_group", ""))
        word = str(item.get("word", ""))
        start = item.get("start", 0)
        end = item.get("end", 0)
        if label not in VALID_LABELS:
            continue

        if label == "职务":
            department_keywords = ["人事部", "财务部", "办公室", "行政部", "技术部", "市场部", "销售部", "研发部", "人事科", "财务科"]
            if word in department_keywords:
                item["entity_group"] = "工作单位"
                continue
            work_content_keywords = ["档案审核", "审核", "会计", "出纳", "管理", "负责"]
            if word in work_content_keywords:
                context_before = text[max(0, start - 20):start]
                if "负责" in context_before or "从事" in context_before:
                    item["_drop"] = True
                    continue

        if label == "身份证号":
            clean_word = re.sub(r'[\s\-_]', '', word)
            is_valid_18 = re.fullmatch(r'[1-9]\d{16}[\dXx]', clean_word) is not None
            is_valid_15 = re.fullmatch(r'[1-9]\d{14}', clean_word) is not None
            if not (is_valid_18 or is_valid_15):
                item["_drop"] = True
                continue

        is_in_work_history = False
        for kw in work_history_keywords:
            pos = text.find(kw)
            if pos != -1 and start > pos:
                is_in_work_history = True
                break

        if family_region_start != -1 and isinstance(start, int) and start >= family_region_start:
            if re.match(r"\d{4}\.\d{1,2}(\.\d{1,2})?", word) or re.match(r"\d{4}\.\d{1,2}", word):
                if label in ["现职时间", "入党时间"]:
                    item["entity_group"] = "出生年月"
            if label == "职务" and word == "党员":
                item["entity_group"] = "政治面貌"

        workunit_keywords = ["公司", "集团", "局", "部", "中心", "研究所", "研究院", "设计院", "人事部", "财务部", "办公室", "厂", "矿", "有限"]
        if any(kw in word for kw in workunit_keywords):
            context_before = text[max(0, start - 50):start]
            work_indicators = ["工作于", "就职", "任职", "现工作", "所在单位", "从事", "工作", "就任"]
            if any(ind in context_before for ind in work_indicators):
                if label in ["全日制教育毕业院校系", "在职教育毕业院校系", "工作单位"]:
                    item["entity_group"] = "工作单位"
                    continue

        company_keywords = ["公司", "集团", "有限", "建筑", "工程", "厂", "矿", "局", "部", "中心"]
        if label in ["全日制教育毕业院校系", "在职教育毕业院校系"]:
            if any(kw in word for kw in company_keywords):
                context_before = text[max(0, start - 150):start]
                work_indicators = ["至今", "工作经历", "简历", "参加工作", "任职", "就职", "年月-", "-至今"]
                if any(ind in context_before for ind in work_indicators):
                    item["entity_group"] = "工作单位"
                    continue

        is_school = any(kw in word for kw in school_keywords)
        if is_school and label in ["工作单位", "全日制教育毕业院校系", "在职教育毕业院校系"]:
            if family_region_start != -1 and isinstance(start, int) and start >= family_region_start:
                item["entity_group"] = "工作单位"
                continue
            if is_in_work_history:
                if any(kw in word for kw in company_keywords):
                    item["entity_group"] = "工作单位"
                    continue
            context_start = max(0, start - 80)
            context_end = min(len(text), end + 80)
            context = text[context_start:context_end]
            is_fulltime = any(kw in context for kw in fulltime_keywords)
            is_parttime = any(kw in context for kw in parttime_keywords)
            if is_parttime and not is_fulltime:
                item["entity_group"] = "在职教育毕业院校系"
            else:
                item["entity_group"] = "全日制教育毕业院校系"

        if label == "籍贯":
            if word.endswith("长治") and isinstance(end, int) and end < len(text):
                if text[end] == "市":
                    item["word"] = word + "市"
                    item["end"] = end + 1
            if word == "西省五台县":
                item["word"] = "山西省五台县"
                item["start"] = start - 1

        if label == "出生年月" and isinstance(start, int) and isinstance(end, int):
            if re.match(r"^\d{4}$", word):
                remaining = text[end:min(len(text), end + 20)]
                match = re.search(r"\.\d{1,2}\.\d{1,2}", remaining)
                if match:
                    full_date = word + match.group(0)
                    item["word"] = full_date
                    item["end"] = end + len(match.group(0))

        if label == "籍贯" and any(k in word for k in ["路", "号", "小区", "街道"]):
            item["entity_group"] = "家庭住址"

        if word in ["本科", "研究生", "专科", "硕士", "博士", "大学本科", "大学专科"]:
            context_start = max(0, start - 40)
            context_end = min(len(text), end + 40)
            context = text[context_start:context_end]
            if "在职" in context or "继续教育" in context:
                item["entity_group"] = "在职教育学历"
            else:
                item["entity_group"] = "全日制教育学历"

        if label == "全日制教育专业":
            position_words = ["财务会计", "会计", "出纳", "工程师", "技术员", "经济师", "会计师", "教师", "医生", "护士"]
            if word in position_words:
                context_before = text[max(0, start - 10):start]
                if "职务" in context_before or "从事何种工作" in context_before:
                    item["entity_group"] = "职务"

        if word == "太原理工大学":
            remaining = text[end:min(len(text), end + 20)]
            if "岩土工程" in remaining:
                full_word = word + "岩土工程"
                item["word"] = full_word
                item["end"] = end + len("岩土工程")
                item["entity_group"] = "全日制教育毕业院校系"

    out = [item for item in out if not item.get("_drop", False)]
    out = [item for item in out if str(item.get("entity_group", "")) in VALID_LABELS]
    return out


def inject_missing_school_name(items: list, text: str) -> list:
    out = [dict(x) for x in items]
    for match in re.finditer(r'毕业院校及专业\s+([^\n]+)', text):
        content = match.group(1)
        content_start = match.start(1)
        has_school = False
        for item in out:
            if item.get("entity_group") == "全日制教育毕业院校系":
                s = item.get("start", 0)
                e = item.get("end", 0)
                if content_start <= s <= content_start + len(content):
                    has_school = True
                    break
        if not has_school:
            school_match = re.search(r'([一-鿿]+?(?:大学|学院|研究院|学校))', content)
            if school_match:
                school_name = school_match.group(1)
                school_start = content_start + school_match.start(1)
                school_end = content_start + school_match.end(1)
                out.append({
                    "entity_group": "全日制教育毕业院校系",
                    "score": 0.999, "word": school_name,
                    "start": school_start, "end": school_end,
                })
    return sorted(out, key=lambda x: int(x.get("start", 0)))


def inject_missing_political_status(items: list, text: str) -> list:
    out = [dict(x) for x in items]
    for match in re.finditer(r'政治面貌[\s,]+([^\s,]+)', text):
        status_start = match.start(1)
        status_end = match.end(1)
        status_word = match.group(1).strip()
        has_political = False
        for item in out:
            if item.get("entity_group") == "政治面貌":
                s = item.get("start", 0)
                e = item.get("end", 0)
                if s <= status_start <= e or (status_start <= s <= status_end):
                    has_political = True
                    break
        if not has_political and status_word in VALID_POLITICAL_STATUS:
            out.append({
                "entity_group": "政治面貌",
                "score": 0.999, "word": status_word,
                "start": status_start, "end": status_end,
            })
    return sorted(out, key=lambda x: int(x.get("start", 0)))


def inject_missing_range_dates(items: list, text: str) -> list:
    if not text:
        return items
    out = [dict(x) for x in items]
    date_labels = {"出生年月", "参加工作时间", "现职时间", "入党时间"}
    ym_pat = re.compile(r"\d{4}\s*年\s*(?:0?[1-9]|1[0-2])\s*月")

    def overlaps_date(s, e):
        for it in out:
            lbl = str(it.get("entity_group", ""))
            is_ = it.get("start")
            ie = it.get("end")
            if lbl not in date_labels or not isinstance(is_, int) or not isinstance(ie, int):
                continue
            if not (e <= is_ or ie <= s):
                return True
        return False

    def nearest_date_score(s, e):
        best_dist = 10 ** 9
        best_score = None
        for it in out:
            lbl = str(it.get("entity_group", ""))
            is_ = it.get("start")
            ie = it.get("end")
            sc = it.get("score")
            if lbl not in date_labels:
                continue
            if not (isinstance(is_, int) and isinstance(ie, int) and isinstance(sc, (int, float))):
                continue
            dist = min(abs(s - is_), abs(e - ie))
            if dist < best_dist:
                best_dist = dist
                best_score = float(sc)
        if best_score is not None:
            return best_score
        return injected_score(out, s, e)

    for m in ym_pat.finditer(text):
        s = m.start()
        e = m.end()
        if overlaps_date(s, e):
            continue
        left = text[max(0, s - 2):s]
        right = text[e:min(len(text), e + 2)]

        label = None
        if re.match(r"^\s*[-—~至到]", right):
            label = "日期"
        elif re.search(r"[-—~至到]\s*$", left):
            label = "日期"

        if label is None:
            continue
        out.append({
            "entity_group": label,
            "score": nearest_date_score(s, e),
            "word": text[s:e], "start": s, "end": e,
        })
    return sorted(out, key=lambda x: int(x.get("start", 0)))


def postprocess_misc_entities(items: list, text: str) -> list:
    fixed = []
    for raw in items:
        it = dict(raw)
        label = str(it.get("entity_group", ""))
        word = str(it.get("word", ""))
        s = it.get("start")
        e = it.get("end")
        if label not in VALID_LABELS:
            continue
        if label in ["全日制教育毕业院校系", "在职教育毕业院校系"]:
            match = re.match(
                r'^([一-鿿]{2,3}[市县区])?([一-鿿]+?(?:大学|学院|学校|研究院|系))$', word
            )
            if match:
                prefix = match.group(1) or ""
                school = match.group(2)
                if prefix and school:
                    it["word"] = school
                    if isinstance(s, int):
                        it["start"] = s + len(prefix)
        if isinstance(s, int) and isinstance(e, int) and 0 <= s <= e <= len(text):
            if label == "毕肄业学校或单位" and s > 0 and any(k in word for k in {"大学", "学院", "学校"}):
                prev = text[s - 1]
                if re.fullmatch(r"[一-鿿]", prev) is not None and prev not in {"于", "在", "到", "从", "由", "与", "和"}:
                    it["start"] = s - 1
                    it["word"] = text[s - 1:e]
        fixed.append(it)
    return fixed


def deduplicate_overlaps(items: list) -> list:
    if not items:
        return items
    label_priority = {
        "全日制教育毕业院校系": 1, "在职教育毕业院校系": 1,
        "全日制教育专业": 2, "在职教育专业": 2,
        "出生年月": 3, "参加工作时间": 3, "入党时间": 3, "现职时间": 3,
        "工作单位": 4, "职务": 5, "政治面貌": 5,
        "全日制教育学历": 5, "全日制教育学位": 5,
        "在职教育学历": 5, "在职教育学位": 5,
        "姓名": 5, "性别": 5, "民族": 5, "籍贯": 5, "出生地": 5, "身份证号": 5,
    }
    kept = []
    for cand in sorted(items, key=lambda x: (int(x.get("start", 0)), int(x.get("end", 0)))):
        cs = cand.get("start")
        ce = cand.get("end")
        cl = str(cand.get("entity_group", ""))
        if not (isinstance(cs, int) and isinstance(ce, int)):
            kept.append(cand)
            continue
        drop = False
        for idx, old in enumerate(kept):
            os_ = old.get("start")
            oe = old.get("end")
            ol = str(old.get("entity_group", ""))
            if not (isinstance(os_, int) and isinstance(oe, int)):
                continue
            if cs == os_ and ce == oe:
                if cand.get("score", 0) > old.get("score", 0):
                    kept[idx] = cand
                drop = True
                break
            if os_ <= cs and ce <= oe:
                drop = True
                break
            if cs <= os_ and oe <= ce:
                old_priority = label_priority.get(ol, 10)
                new_priority = label_priority.get(cl, 10)
                if new_priority < old_priority:
                    kept[idx] = cand
                drop = True
                break
            if cs < oe and ce > os_:
                old_priority = label_priority.get(ol, 10)
                new_priority = label_priority.get(cl, 10)
                if new_priority < old_priority:
                    kept[idx] = cand
                drop = True
                break
        if not drop:
            kept.append(cand)
    return sorted(kept, key=lambda x: int(x.get("start", 0)))


def postprocess_org_entities(items: list, text: str) -> list:
    if not items:
        return items
    org_labels = {"工作单位"}
    merged = []
    i = 0
    while i < len(items):
        cur = dict(items[i])
        cs = cur.get("start")
        ce = cur.get("end")
        cl = str(cur.get("entity_group", ""))
        if not (isinstance(cs, int) and isinstance(ce, int)):
            merged.append(cur)
            i += 1
            continue
        if cl in org_labels:
            j = i + 1
            best_start = cs
            best_end = ce
            best_score = float(cur.get("score", 0.0))
            while j < len(items):
                nxt = items[j]
                ns = nxt.get("start")
                ne = nxt.get("end")
                nl = str(nxt.get("entity_group", ""))
                if not (isinstance(ns, int) and isinstance(ne, int)):
                    break
                if nl not in org_labels:
                    break
                gap = text[best_end:ns]
                if len(gap) <= 2 and re.fullmatch(r"[\s,，;；:：\|]*", gap) is not None:
                    best_end = ne
                    best_score = max(best_score, float(nxt.get("score", 0.0)))
                    j += 1
                    continue
                break
            if best_end != ce:
                cur["start"] = best_start
                cur["end"] = best_end
                cur["word"] = text[best_start:best_end]
                cur["score"] = best_score
                i = j
            else:
                i += 1
        else:
            i += 1
        merged.append(cur)
    return merged


# ============== 主入口函数 ==============

def postprocess_ner_entities(raw_entities: list, full_text: str, score_threshold: float = 0.98) -> list:
    """对 NER pipeline 的原始输出进行完整的后处理。

    Args:
        raw_entities: transformers pipeline 返回的实体列表
        full_text: OCR 全文
        score_threshold: 置信度阈值（低于此值的实体被丢弃）

    Returns:
        后处理后的实体列表，每个元素包含 entity_group, word, start, end, score
    """
    # 初步过滤
    cleaned_items = []
    for ent in raw_entities:
        score = ent.get("score", 0.0)
        if score < score_threshold:
            continue
        ent_dict = dict(ent)
        start = ent_dict.get("start")
        end = ent_dict.get("end")
        if isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(full_text):
            ent_dict["word"] = full_text[start:end]
        elif "word" in ent_dict and isinstance(ent_dict["word"], str):
            ent_dict["word"] = normalize_word(ent_dict["word"])
        if isinstance(ent_dict.get("word"), str):
            ent_dict["word"] = clean_entity_text(ent_dict["word"])
        label = str(ent_dict.get("entity_group", ""))
        word = str(ent_dict.get("word", ""))
        if should_drop_entity(label, word, full_text,
                               start if isinstance(start, int) else None,
                               end if isinstance(end, int) else None):
            continue
        ent_dict["score"] = score
        cleaned_items.append(ent_dict)

    cleaned_items.sort(key=lambda x: int(x.get("start", 0)))
    cleaned_items = fix_adjacent_split_entities(cleaned_items, full_text)
    cleaned_items = fix_incomplete_dates(cleaned_items, full_text)
    cleaned_items = context_relabel(cleaned_items, full_text)
    cleaned_items = inject_missing_school_name(cleaned_items, full_text)
    cleaned_items = inject_missing_political_status(cleaned_items, full_text)
    cleaned_items = inject_missing_range_dates(cleaned_items, full_text)
    cleaned_items = postprocess_misc_entities(cleaned_items, full_text)
    cleaned_items = deduplicate_overlaps(cleaned_items)
    cleaned_items = postprocess_org_entities(cleaned_items, full_text)

    # 最终过滤
    cleaned_items = [it for it in cleaned_items if it.get("entity_group") in VALID_LABELS]
    cleaned_items = [
        it for it in cleaned_items
        if not (it.get("entity_group") in ["全日制教育学历", "在职教育学历"] and it.get("word") == "大学")
    ]
    return cleaned_items


def predict_fields_ner(ner_pipeline, text: str, fields: list, score_threshold: float = 0.98) -> dict:
    """使用 NER pipeline 从文本中提取字段，返回 {field: value} 字典。"""
    max_len = 512
    chunk_size = 500
    overlap = 50

    if len(text) <= max_len:
        raw_entities = ner_pipeline(text)
    else:
        raw_entities = []
        start_offset = 0
        while start_offset < len(text):
            end_offset = min(start_offset + chunk_size, len(text))
            chunk = text[start_offset:end_offset]
            try:
                chunk_entities = ner_pipeline(chunk)
            except Exception:
                start_offset += chunk_size - overlap
                continue
            for ent in chunk_entities:
                ent = dict(ent)
                ent['start'] = start_offset + ent['start']
                ent['end'] = start_offset + ent['end']
                raw_entities.append(ent)
            start_offset += chunk_size - overlap

        raw_entities.sort(key=lambda x: (x['start'], -x['end']))
        merged = []
        for ent in raw_entities:
            if not merged:
                merged.append(ent)
                continue
            last = merged[-1]
            if ent['start'] <= last['end'] and ent['entity_group'] == last['entity_group']:
                last['end'] = max(last['end'], ent['end'])
                if ent.get('score', 0) > last.get('score', 0):
                    last['score'] = ent['score']
                    last['word'] = text[last['start']:last['end']]
            else:
                merged.append(ent)
        raw_entities = merged

    cleaned_items = postprocess_ner_entities(raw_entities, text, score_threshold)

    best = {f: "" for f in fields}
    for ent in cleaned_items:
        label = ent.get("entity_group")
        if label in best:
            best[label] = ent.get("word", "")
    return best
