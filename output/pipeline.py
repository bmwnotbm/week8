from pathlib import Path
import json
import numpy as np
import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", None)

DATA = Path(__file__).parent / "data"
OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

# เก็บทุกเหตุการณ์ cleaning/exclusion ไว้ที่นี่ เพื่อสร้าง data_quality_report.csv
# ทำให้ทุกแถวที่ถูกคัดออก/แก้ไข สามารถตรวจสอบย้อนกลับได้ (traceability)
dq_log = []


def log_dq(stage, check, n_rows, note):
    """บันทึกเหตุการณ์คุณภาพข้อมูลหนึ่งรายการลง dq_log"""
    dq_log.append({"stage": stage, "check": check, "row_count": int(n_rows), "note": note})
    print(f"  [{stage}] {check}: {n_rows} แถว — {note}")


def profile(df, name):
    """แสดง shape, dtypes, missing, duplicate ของ DataFrame แบบสรุป"""
    print(f"\n--- PROFILE: {name} ---")
    print("shape:", df.shape)
    print("dtypes:\n", df.dtypes)
    print("missing values:\n", df.isna().sum())
    print("duplicate rows (all columns):", df.duplicated().sum())
    print("sample:\n", df.head(3))


# =========================================================================
# 5.1 EXTRACT & PROFILE — อ่านทุกไฟล์ และสำรวจคุณภาพข้อมูล "ก่อน" แก้ไขใดๆ
# =========================================================================
print("=" * 80)
print("STEP 1: EXTRACT & PROFILE (RAW DATA)")
print("=" * 80)

orders_01_raw = pd.read_csv(DATA / "orders_2026_01.csv")
orders_02_raw = pd.read_csv(DATA / "orders_2026_02.csv")
customers_raw = pd.read_csv(DATA / "customers_crm.csv")
products_raw = pd.read_excel(DATA / "product_master.xlsx")

with open(DATA / "payments.json", "r", encoding="utf-8") as f:
    payments_raw_json = json.load(f)
payments_raw = pd.json_normalize(payments_raw_json)  # แตก nested {"payment": {"method","status"}}

profile(orders_01_raw, "orders_2026_01 (raw)")
profile(orders_02_raw, "orders_2026_02 (raw)")
profile(customers_raw, "customers_crm (raw)")
profile(products_raw, "product_master (raw)")
profile(payments_raw, "payments (raw, flattened)")

# เก็บ snapshot "ก่อน" ไว้เทียบกับ "หลัง" ตอนสรุปท้ายรายงาน
before_summary = {
    "orders_2026_01_rows": len(orders_01_raw),
    "orders_2026_02_rows": len(orders_02_raw),
    "orders_total_raw": len(orders_01_raw) + len(orders_02_raw),
    "customers_rows": len(customers_raw),
    "customers_duplicate_rows": int(customers_raw.duplicated().sum()),
    "customers_missing_email": int(customers_raw["email"].isna().sum()),
    "products_rows": len(products_raw),
    "payments_rows": len(payments_raw),
    "payments_duplicate_order_id": int(payments_raw["order_id"].duplicated().sum()),
}

# =========================================================================
# 5.2 COMBINE ORDERS — ปรับ schema เดือน ก.พ. ให้ตรงกับเดือน ม.ค. แล้ว concat
# =========================================================================
print("\n" + "=" * 80)
print("STEP 2: COMBINE ORDERS (SCHEMA ALIGNMENT + CONCAT)")
print("=" * 80)

# ม.ค. : order_id, order_date, customer_id, product_id, quantity, unit_price, discount, channel
# ก.พ. : order_id, ordered_at, customer_id, product_id, qty,      unit_price, discount_pct, channel
#   -> ชื่อคอลัมน์ต่างกัน (ordered_at/qty/discount_pct)
#   -> discount_pct เป็น string แบบ "5%" ต้องแปลงเป็นสัดส่วน 0-1
#   -> ordered_at เป็นรูปแบบวันที่ DD/MM/YYYY HH:MM ต่างจาก ม.ค. ที่เป็น YYYY-MM-DD HH:MM:SS

orders_02_aligned = orders_02_raw.rename(
    columns={"ordered_at": "order_date", "qty": "quantity", "discount_pct": "discount"}
).copy()

# แปลง "5%" -> 0.05 (ตัด % ออกแล้วหารด้วย 100)
orders_02_aligned["discount"] = (
    orders_02_aligned["discount"].astype(str).str.rstrip("%").astype(float) / 100.0
)

# ปรับรูปแบบวันที่ให้เป็น datetime ตรงกันทั้งสองเดือนก่อน concat
orders_01_aligned = orders_01_raw.copy()
orders_01_aligned["order_date"] = pd.to_datetime(orders_01_aligned["order_date"], errors="coerce")
orders_02_aligned["order_date"] = pd.to_datetime(
    orders_02_aligned["order_date"], format="%d/%m/%Y %H:%M", errors="coerce"
)

# จัดลำดับคอลัมน์ให้ตรงกันก่อน concat
col_order = ["order_id", "order_date", "customer_id", "product_id", "quantity", "unit_price", "discount", "channel"]
orders_01_aligned = orders_01_aligned[col_order]
orders_02_aligned = orders_02_aligned[col_order]

orders_raw_combined = pd.concat([orders_01_aligned, orders_02_aligned], ignore_index=True)
n_after_concat = len(orders_raw_combined)
log_dq(
    "combine_orders",
    "concat ม.ค.+ก.พ.",
    n_after_concat,
    f"รวม orders_2026_01 ({len(orders_01_aligned)}) + orders_2026_02 ({len(orders_02_aligned)}) = {n_after_concat} แถว",
)
print(f"\nแถวรวมหลัง concat: {n_after_concat}")

# =========================================================================
# 5.3 TRANSFORM — แปลงชนิดข้อมูล, ทำความสะอาด, standardize, dedup
# =========================================================================
print("\n" + "=" * 80)
print("STEP 3: TRANSFORM (CLEAN / STANDARDIZE / DEDUPLICATE)")
print("=" * 80)

orders = orders_raw_combined.copy()

# --- ชนิดข้อมูล ---
orders["quantity"] = pd.to_numeric(orders["quantity"], errors="coerce")
orders["unit_price"] = pd.to_numeric(orders["unit_price"], errors="coerce")
orders["discount"] = pd.to_numeric(orders["discount"], errors="coerce")

# --- ลบ exact-duplicate rows ก่อน (ป้องกัน order_id ซ้ำเพราะข้อมูลถูกคีย์ซ้ำทั้งแถว) ---
n_before_exact_dedup = len(orders)
exact_dupe_mask = orders.duplicated(keep="last")
n_exact_dupes = int(exact_dupe_mask.sum())
if n_exact_dupes:
    log_dq("clean_orders", "exact duplicate rows", n_exact_dupes, "แถวซ้ำทุกคอลัมน์ทุกประการ — ลบออก เก็บแถวหลังสุด")
orders = orders[~exact_dupe_mask].copy()

# --- กติกา: order_id ต้องมีเพียง 1 แถว หากซ้ำให้เก็บ "ล่าสุดตามลำดับที่ปรากฏ" (keep='last') ---
n_orderid_dupe = int(orders["order_id"].duplicated(keep="last").sum())
if n_orderid_dupe:
    log_dq(
        "clean_orders",
        "duplicate order_id (schema/value ต่างกัน)",
        n_orderid_dupe,
        "order_id ซ้ำแต่ค่าบางคอลัมน์ต่างกัน — เก็บแถวที่ปรากฏหลังสุดตามกติกาธุรกิจ",
    )
orders = orders.drop_duplicates(subset="order_id", keep="last").copy()

# --- validate business rules เชิงตัวเลข: quantity>0, unit_price>0, 0<=discount<=1 ---
missing_price_mask = orders["unit_price"].isna()
n_missing_price = int(missing_price_mask.sum())
if n_missing_price:
    log_dq("clean_orders", "unit_price ว่าง (missing)", n_missing_price, "ไม่สามารถคำนวณ net_sales ได้ — คัดออก")

bad_qty_mask = (~missing_price_mask) & (orders["quantity"] <= 0)
n_bad_qty = int(bad_qty_mask.sum())
if n_bad_qty:
    log_dq("clean_orders", "quantity <= 0", n_bad_qty, "ผิดกติกาธุรกิจ (quantity ต้อง > 0) — คัดออก")

bad_price_mask = (~missing_price_mask) & (orders["unit_price"] <= 0)
n_bad_price = int(bad_price_mask.sum())
if n_bad_price:
    log_dq("clean_orders", "unit_price <= 0", n_bad_price, "ผิดกติกาธุรกิจ (unit_price ต้อง > 0) — คัดออก")

bad_discount_mask = (orders["discount"] < 0) | (orders["discount"] > 1) | orders["discount"].isna()
n_bad_discount = int(bad_discount_mask.sum())
if n_bad_discount:
    log_dq("clean_orders", "discount นอกช่วง [0,1] หรือว่าง", n_bad_discount, "ผิดกติกาธุรกิจ — คัดออก")

drop_mask = missing_price_mask | bad_qty_mask | bad_price_mask | bad_discount_mask
orders_clean = orders[~drop_mask].copy()

print(f"\norders: {n_before_exact_dedup} -> หลัง dedup+validate ตัวเลข -> {len(orders_clean)} แถว")

# --- Customers: standardize ---
customers = customers_raw.copy()

# ลบ exact-duplicate rows (customer_id ซ้ำทั้งแถว)
n_cust_before = len(customers)
cust_exact_dupe = customers.duplicated(subset="customer_id", keep="first")
n_cust_dupe = int(cust_exact_dupe.sum())
if n_cust_dupe:
    log_dq("clean_customers", "duplicate customer_id (exact)", n_cust_dupe, "แถวซ้ำ customer_id เดิมทุกค่า — เก็บแถวแรก")
customers = customers[~cust_exact_dupe].copy()

# email -> lower-case + trim; ถ้าว่างให้คงเป็น NaN (ไม่ทิ้งลูกค้า เพราะยังใช้ merge ด้วย customer_id ได้)
n_missing_email = int(customers["email"].isna().sum())
if n_missing_email:
    log_dq("clean_customers", "email ว่าง (missing)", n_missing_email, "คงไว้ในตาราง (merge ใช้ customer_id) แต่ไม่มีอีเมลสำหรับติดต่อ")
customers["email"] = customers["email"].str.strip().str.lower()

# province -> ชื่อมาตรฐาน (รวมชื่อไทย/อังกฤษ/คำย่อ/ตัวสะกดผิดให้เป็นค่าเดียว)
province_map = {
    "ชลบุรี": "ชลบุรี",
    "chonburi": "ชลบุรี",
    "ขอนแก่น": "ขอนแก่น",
    "ขอนเเก่น": "ขอนแก่น",  # สะกดด้วยสระ เ-แ ซ้ำ (encoding ต่างกัน)
    "กรุงเทพมหานคร": "กรุงเทพมหานคร",
    "bangkok": "กรุงเทพมหานคร",
    "กทม.": "กรุงเทพมหานคร",
    "ระยอง": "ระยอง",
    "rayong": "ระยอง",
    "phuket": "ภูเก็ต",
    "ภูเก็ต": "ภูเก็ต",
    "chiang mai": "เชียงใหม่",
    "เชียงใหม่": "เชียงใหม่",
}
n_province_before = customers["province"].nunique()
customers["province_raw"] = customers["province"]
customers["province"] = (
    customers["province"].str.strip().str.lower().map(province_map).fillna(customers["province"].str.strip())
)
n_province_after = customers["province"].nunique()
log_dq(
    "clean_customers",
    "standardize province",
    len(customers),
    f"รวมชื่อจังหวัดจาก {n_province_before} รูปแบบ เหลือ {n_province_after} ค่ามาตรฐาน",
)
customers["full_name"] = customers["full_name"].str.strip()
customers = customers.drop(columns=["province_raw"])
print(f"\ncustomers: {n_cust_before} -> หลัง dedup -> {len(customers)} แถว")

# --- Products: standardize ---
products = products_raw.copy()
products["product_name"] = products["product_name"].str.strip()
products["category"] = products["category"].str.strip()
n_prod_dupe = int(products.duplicated(subset="product_id").sum())
if n_prod_dupe:
    log_dq("clean_products", "duplicate product_id", n_prod_dupe, "เก็บแถวแรก")
products = products.drop_duplicates(subset="product_id", keep="first").copy()

# --- Payments: standardize ---
payments = payments_raw.rename(columns={"payment.method": "payment_method", "payment.status": "payment_status"}).copy()
payments["paid_at"] = pd.to_datetime(payments["paid_at"], errors="coerce")

n_pay_before = len(payments)
pay_dupe = payments.duplicated(subset="order_id", keep="last")
n_pay_dupe = int(pay_dupe.sum())
if n_pay_dupe:
    log_dq("clean_payments", "duplicate order_id ใน payments", n_pay_dupe, "order_id เดียวกันมีหลาย payment event — เก็บ event ล่าสุด")
payments = payments[~pay_dupe].copy()
print(f"\npayments: {n_pay_before} -> หลัง dedup -> {len(payments)} แถว")

# =========================================================================
# 5.4 INTEGRATE & VALIDATE — merge พร้อม validate= และ indicator=True
# =========================================================================
print("\n" + "=" * 80)
print("STEP 4: INTEGRATE (MERGE) & VALIDATE")
print("=" * 80)

# --- merge กับ customers (many orders : 1 customer) ---
merged = orders_clean.merge(
    customers[["customer_id", "full_name", "email", "province", "signup_date"]],
    on="customer_id",
    how="left",
    validate="m:1",
    indicator="_merge_customer",
)
cust_match_counts = merged["_merge_customer"].value_counts()
n_customer_unmatched = int((merged["_merge_customer"] == "left_only").sum())
log_dq(
    "integrate",
    "orders x customers (m:1)",
    n_customer_unmatched,
    f"customer_id ไม่พบใน Master Data: {n_customer_unmatched} แถว | matched: {int(cust_match_counts.get('both', 0))} แถว",
)

# --- merge กับ products (many orders : 1 product) ---
merged = merged.merge(
    products[["product_id", "product_name", "category", "standard_price", "active_flag"]],
    on="product_id",
    how="left",
    validate="m:1",
    indicator="_merge_product",
)
prod_match_counts = merged["_merge_product"].value_counts()
n_product_unmatched = int((merged["_merge_product"] == "left_only").sum())
log_dq(
    "integrate",
    "orders x products (m:1)",
    n_product_unmatched,
    f"product_id ไม่พบใน Master Data: {n_product_unmatched} แถว | matched: {int(prod_match_counts.get('both', 0))} แถว",
)

# --- merge กับ payments (1 order : 1 payment event หลัง dedup) ---
merged = merged.merge(
    payments[["order_id", "payment_id", "payment_method", "payment_status", "paid_at"]],
    on="order_id",
    how="left",
    validate="1:1",
    indicator="_merge_payment",
)
pay_match_counts = merged["_merge_payment"].value_counts()
n_payment_unmatched = int((merged["_merge_payment"] == "left_only").sum())
log_dq(
    "integrate",
    "orders x payments (1:1)",
    n_payment_unmatched,
    f"order_id ไม่พบ payment event: {n_payment_unmatched} แถว | matched: {int(pay_match_counts.get('both', 0))} แถว",
)

# --- คัดเฉพาะแถวที่จับคู่ครบทั้ง customer และ product (ตามกติกา: ลูกค้าและสินค้าต้องพบใน Master Data) ---
fully_matched_mask = (merged["_merge_customer"] == "both") & (merged["_merge_product"] == "both")
n_ref_integrity_dropped = int((~fully_matched_mask).sum())
if n_ref_integrity_dropped:
    log_dq(
        "integrate",
        "referential integrity (customer & product ต้อง match)",
        n_ref_integrity_dropped,
        "customer_id หรือ product_id ไม่พบใน Master Data — คัดออกจาก fact table",
    )

fact = merged[fully_matched_mask].drop(columns=["_merge_customer", "_merge_product", "_merge_payment"]).copy()

# =========================================================================
# 5.4b คำนวณ net_sales และกำหนดว่าธุรกรรมใด "ใช้ได้จริง" (นับเป็นยอดขาย)
# =========================================================================
fact["net_sales"] = fact["quantity"] * fact["unit_price"] * (1 - fact["discount"])
fact["is_paid_sale"] = fact["payment_status"] == "PAID"

n_paid = int(fact["is_paid_sale"].sum())
n_not_paid = int((~fact["is_paid_sale"]).sum())
log_dq(
    "validate",
    "payment_status != PAID (ไม่นับเป็นยอดขาย)",
    n_not_paid,
    "รวมอยู่ใน fact_sales เพื่อการตรวจสอบย้อนกลับ แต่ถูกคัดออกจากยอดขายสุทธิ (ใช้ is_paid_sale=False กรองใน summary)",
)
log_dq("validate", "ธุรกรรมที่นับเป็นยอดขายจริง (PAID)", n_paid, f"ยอดขายสุทธิรวม = {fact.loc[fact['is_paid_sale'],'net_sales'].sum():,.2f} บาท")

print(f"\nfact_sales (matched ครบ customer+product): {len(fact)} แถว")
print(f"  - PAID (นับเป็นยอดขาย): {n_paid} แถว")
print(f"  - ไม่ใช่ PAID (FAILED/REFUNDED/ไม่มี payment): {n_not_paid} แถว")

# =========================================================================
# 5.5 LOAD — dim_customer, dim_product, fact_sales, data_quality_report
# =========================================================================
print("\n" + "=" * 80)
print("STEP 5: LOAD (WRITE OUTPUT FILES)")
print("=" * 80)

dim_customer = customers[["customer_id", "full_name", "email", "province", "signup_date"]].reset_index(drop=True)
dim_product = products[["product_id", "product_name", "category", "standard_price", "active_flag"]].reset_index(drop=True)

fact_sales = fact[
    [
        "order_id",
        "order_date",
        "customer_id",
        "product_id",
        "quantity",
        "unit_price",
        "discount",
        "net_sales",
        "channel",
        "province",
        "category",
        "payment_id",
        "payment_method",
        "payment_status",
        "paid_at",
        "is_paid_sale",
    ]
].sort_values("order_id").reset_index(drop=True)

dim_customer.to_csv(OUTPUT / "dim_customer.csv", index=False, encoding="utf-8-sig")
dim_product.to_csv(OUTPUT / "dim_product.csv", index=False, encoding="utf-8-sig")
fact_sales.to_csv(OUTPUT / "fact_sales.csv", index=False, encoding="utf-8-sig")

dq_report = pd.DataFrame(dq_log)
dq_report.to_csv(OUTPUT / "data_quality_report.csv", index=False, encoding="utf-8-sig")

print("เขียนไฟล์: dim_customer.csv, dim_product.csv, fact_sales.csv, data_quality_report.csv")

# =========================================================================
# 5.6 ANALYZE — summary_by_province, summary_by_category
# =========================================================================
print("\n" + "=" * 80)
print("STEP 6: ANALYZE")
print("=" * 80)

paid_sales = fact_sales[fact_sales["is_paid_sale"]].copy()

summary_by_province = (
    paid_sales.groupby("province", as_index=False)
    .agg(net_sales=("net_sales", "sum"), n_transactions=("order_id", "count"), avg_order_value=("net_sales", "mean"))
    .sort_values("net_sales", ascending=False)
    .reset_index(drop=True)
)

summary_by_category = (
    paid_sales.groupby("category", as_index=False)
    .agg(net_sales=("net_sales", "sum"), n_transactions=("order_id", "count"), avg_order_value=("net_sales", "mean"))
    .sort_values("net_sales", ascending=False)
    .reset_index(drop=True)
)

summary_by_province.to_csv(OUTPUT / "summary_by_province.csv", index=False, encoding="utf-8-sig")
summary_by_category.to_csv(OUTPUT / "summary_by_category.csv", index=False, encoding="utf-8-sig")

print("\nsummary_by_province:\n", summary_by_province)
print("\nsummary_by_category:\n", summary_by_category)

# =========================================================================
# สรุปคุณภาพข้อมูล ก่อน / หลัง Integration
# =========================================================================
after_summary = {
    "orders_combined_raw": n_after_concat,
    "orders_after_dedup_and_validation": len(orders_clean),
    "customers_after_dedup": len(customers),
    "products_after_dedup": len(products),
    "payments_after_dedup": len(payments),
    "fact_sales_rows": len(fact_sales),
    "paid_transactions": n_paid,
    "total_net_sales_paid": round(float(paid_sales["net_sales"].sum()), 2),
}

print("\n" + "=" * 80)
print("DATA QUALITY: BEFORE vs AFTER INTEGRATION")
print("=" * 80)
print("BEFORE (raw):")
for k, v in before_summary.items():
    print(f"  {k}: {v}")
print("\nAFTER (integrated & validated):")
for k, v in after_summary.items():
    print(f"  {k}: {v}")

# =========================================================================
# คำตอบคำถามวิเคราะห์ 6 ข้อ
# =========================================================================
print("\n" + "=" * 80)
print("คำถามวิเคราะห์ 6 ข้อ")
print("=" * 80)

q1 = (
    f"Q1: หลังรวมไฟล์ orders (concat ม.ค.+ก.พ.) มี {n_after_concat} แถว "
    f"หลังลบ duplicate (exact duplicate {n_exact_dupes} แถว + duplicate order_id {n_orderid_dupe} แถว) "
    f"เหลือ {n_after_concat - n_exact_dupes - n_orderid_dupe} แถว ก่อนกรองกติกาตัวเลขอื่น "
    f"(ตัวเลขสุดท้ายหลัง validate ครบทุกกติกา = {len(orders_clean)} แถว)"
)
q2 = (
    f"Q2: customer_id ไม่พบใน Master Data = {n_customer_unmatched} แถว, "
    f"product_id ไม่พบใน Master Data = {n_product_unmatched} แถว "
    f"(นับจาก {len(orders_clean)} แถวที่ผ่านการทำความสะอาดแล้ว)"
)
q3 = (
    f"Q3: ยอดขายที่ใช้ได้จริง (payment_status = PAID) มี {n_paid} ธุรกรรม "
    f"ยอดขายสุทธิรวม = {paid_sales['net_sales'].sum():,.2f} บาท"
)
top_province = summary_by_province.iloc[0]
q4 = f"Q4: จังหวัดที่มียอดขายสุทธิสูงสุดคือ '{top_province['province']}' = {top_province['net_sales']:,.2f} บาท"
top_category = summary_by_category.iloc[0]
q5 = f"Q5: หมวดสินค้าที่มียอดขายสุทธิสูงสุดคือ '{top_category['category']}' = {top_category['net_sales']:,.2f} บาท"
q6 = (
    "Q6: หากสลับลำดับเป็น merge ก่อน cleaning ผลลัพธ์ที่ได้จะมีคุณภาพต่ำลงและความเชื่อมั่นลดลง เพราะ:\n"
    "     (1) แถวซ้ำ (duplicate order_id/exact-duplicate) และ discount/quantity/unit_price ที่ผิดกติกา "
    "ยังไม่ถูกกรอง ทำให้ merge สร้างแถวผลลัพธ์ที่ไม่ถูกต้อง (เช่น net_sales คำนวณจากข้อมูลขยะ) และอาจทำให้ยอดขายสุทธิเพี้ยนได้ทั้งสูงหรือต่ำเกินจริง\n"
    "     (2) province/email ที่ยังไม่ standardize จะทำให้ summary_by_province แตกเป็นหลายกลุ่มย่อยที่จริงๆ คือจังหวัดเดียวกัน "
    "(เช่น 'Bangkok' กับ 'กรุงเทพมหานคร' จะถูกนับแยกกัน) ทำให้ผลวิเคราะห์คลาดเคลื่อน\n"
    "     (3) validate='m:1'/'1:1' และ referential-integrity check จะตรวจจับปัญหาได้ยากขึ้น เพราะ dirty data "
    "อาจบังเอิญ merge ผ่านแบบผิดๆ (เช่น key ซ้ำที่ยังไม่ dedup ทำให้เกิด duplication แบบ m:m โดยไม่ได้ตั้งใจ)\n"
    "     สรุป: การ clean ก่อน merge (validate-first) ทำให้ผลลัพธ์ตรวจสอบย้อนกลับได้และเชื่อถือได้มากกว่า merge ก่อน clean"
)

for q in [q1, q2, q3, q4, q5, q6]:
    print("\n" + q)

with open(OUTPUT / "analysis_answers.txt", "w", encoding="utf-8") as f:
    f.write("TechTrove — คำตอบคำถามวิเคราะห์ 6 ข้อ\n")
    f.write("=" * 60 + "\n\n")
    for q in [q1, q2, q3, q4, q5, q6]:
        f.write(q + "\n\n")
    f.write("=" * 60 + "\n")
    f.write("สรุปคุณภาพข้อมูล ก่อน/หลัง Integration\n")
    f.write("=" * 60 + "\n\nBEFORE (raw):\n")
    for k, v in before_summary.items():
        f.write(f"  {k}: {v}\n")
    f.write("\nAFTER (integrated & validated):\n")
    for k, v in after_summary.items():
        f.write(f"  {k}: {v}\n")

print("\nเขียนไฟล์: summary_by_province.csv, summary_by_category.csv, analysis_answers.txt")

# =========================================================================
# CHALLENGE (+2): validate_data() และ data-quality funnel
# =========================================================================
print("\n" + "=" * 80)
print("CHALLENGE: validate_data() + Data-Quality Funnel")
print("=" * 80)


def validate_data(df: pd.DataFrame) -> None:
    """
    ตรวจสอบ fact table ด้วย assert:
      - order_id ต้อง unique (uniqueness)
      - customer_id, product_id ทุกแถวต้องพบใน dim_customer / dim_product (referential integrity)
      - quantity>0, unit_price>0, 0<=discount<=1 (ค่าที่อยู่นอกช่วง)
    หากพบปัญหาจะ raise AssertionError พร้อมข้อความบอกจำนวนแถวที่ผิด
    """
    dup = df["order_id"].duplicated().sum()
    assert dup == 0, f"พบ order_id ซ้ำ {dup} แถว (ละเมิด uniqueness)"

    bad_cust = (~df["customer_id"].isin(dim_customer["customer_id"])).sum()
    assert bad_cust == 0, f"พบ customer_id ที่ไม่อยู่ใน dim_customer {bad_cust} แถว (ละเมิด referential integrity)"

    bad_prod = (~df["product_id"].isin(dim_product["product_id"])).sum()
    assert bad_prod == 0, f"พบ product_id ที่ไม่อยู่ใน dim_product {bad_prod} แถว (ละเมิด referential integrity)"

    out_of_range = ((df["quantity"] <= 0) | (df["unit_price"] <= 0) | (df["discount"] < 0) | (df["discount"] > 1)).sum()
    assert out_of_range == 0, f"พบค่าที่อยู่นอกช่วงที่กำหนด {out_of_range} แถว"

    print("validate_data(fact_sales): PASSED — uniqueness, referential integrity, value-range ทั้งหมดผ่าน")


validate_data(fact_sales)

# Funnel: raw -> deduplicated -> matched (customer+product) -> paid sales
funnel_stages = ["raw", "deduplicated", "matched (customer+product)", "paid sales"]
funnel_counts = [
    n_after_concat,                                   # raw (หลัง concat, ก่อน clean ใดๆ)
    n_after_concat - n_exact_dupes - n_orderid_dupe,   # deduplicated
    len(fact),                                         # matched customer+product
    n_paid,                                            # paid sales
]
funnel_df = pd.DataFrame({"stage": funnel_stages, "row_count": funnel_counts})
funnel_df.to_csv(OUTPUT / "data_quality_funnel.csv", index=False, encoding="utf-8-sig")
print("\nData-Quality Funnel:\n", funnel_df)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(funnel_stages[::-1], funnel_counts[::-1], color="#2f6fed")
    ax.set_xlabel("Row count")
    ax.set_title("Data Quality Funnel: raw -> deduplicated -> matched -> paid sales")
    for bar, val in zip(bars, funnel_counts[::-1]):
        ax.text(val, bar.get_y() + bar.get_height() / 2, f" {val:,}", va="center")
    plt.tight_layout()
    fig.savefig(OUTPUT / "data_quality_funnel.png", dpi=150)
    print("บันทึกกราฟ: data_quality_funnel.png")
except ImportError:
    print("(matplotlib ไม่ได้ติดตั้ง — ข้ามการสร้างกราฟ funnel, มีเฉพาะ data_quality_funnel.csv)")

print("\nDone. ตรวจผลลัพธ์ทั้งหมดได้ในโฟลเดอร์:", OUTPUT.resolve())
