# Demo: scripted auditor on dev episode 5 (EP-434114113CE2)

Task: You are auditing the accounts-receivable books of Kaselgar Engineering S.A. for the period 2025-12-01T00:00:00+05:30 .. 2025-12-31T23:59:59+05:30 (company timezone +05:30, base currency INR). Records are read-only and reachable only through the tools. Find booking discrepancies of the types listed in policy.finding_types (there may be none) and submit them with supporting evidence ids. Amounts are integer minor units; never compare different currencies without conversion. You have 100 steps including submit; every step before submit costs a little reward.

- step   1: list_records(invoices, cursor=None) -> 50 records; reward=0.0
- step   2: list_records(invoices, cursor=invoices@50) -> 39 records; reward=0.0
- step   3: list_records(payments, cursor=None) -> 50 records; reward=0.0
- step   4: list_records(payments, cursor=payments@50) -> 2 records; reward=0.0
- step   5: list_records(ledger, cursor=None) -> 50 records; reward=0.0
- step   6: list_records(ledger, cursor=ledger@50) -> 50 records; reward=0.0
- step   7: list_records(ledger, cursor=ledger@100) -> 50 records; reward=0.0
- step   8: list_records(ledger, cursor=ledger@150) -> 50 records; reward=0.0
- step   9: list_records(ledger, cursor=ledger@200) -> 50 records; reward=0.0
- step  10: list_records(ledger, cursor=ledger@250) -> 50 records; reward=0.0
- step  11: list_records(ledger, cursor=ledger@300) -> 50 records; reward=0.0
- step  12: list_records(ledger, cursor=ledger@350) -> 50 records; reward=0.0
- step  13: list_records(ledger, cursor=ledger@400) -> 3 records; reward=0.0
- step  14: get_document(DOC-K0AGMHDF) -> 1 line items; reward=0.0
- step  15: get_document(DOC-F7T4DQWZ) -> 1 line items; reward=0.0
- step  16: get_document(DOC-QCESRWV2) -> 1 line items; reward=0.0
- step  17: get_document(DOC-4XPEVB99) -> 1 line items; reward=0.0
- step  18: get_document(DOC-YB0JBRR7) -> 1 line items; reward=0.0
- step  19: get_document(DOC-G0RWDFS6) -> 1 line items; reward=0.0
- step  20: get_document(DOC-3HVJ8HE6) -> 1 line items; reward=0.0
- step  21: get_document(DOC-YF7CMFQG) -> 1 line items; reward=0.0
- step  22: get_document(DOC-ZFS5EJF7) -> 1 line items; reward=0.0
- step  23: get_document(DOC-YQ3TVZYG) -> 3 line items; reward=0.0
- step  24: get_document(DOC-MQNC44ZY) -> 3 line items; reward=0.0
- step  25: get_record(fx, FX:2025-12-26:UZS:INR) -> rate 0.0064898515; reward=0.0
- step  26: get_record(fx, FX:2025-12-31:CNY:INR) -> rate 10.803349; reward=0.0
- step  27: get_record(fx, FX:2025-12-17:CNY:INR) -> rate 10.641695; reward=0.0
- step  28: get_record(fx, FX:2025-12-17:UZS:INR) -> rate 0.0064407845; reward=0.0
- step  29: get_record(fx, FX:2025-12-26:CNY:INR) -> rate 10.869474; reward=0.0
- step  30: submit(4 findings); reward=0.942
    - duplicate_invoice ['ST002546', 'ST002550'] 80460000 INR evidence=['DOC-YQ3TVZYG', 'DOC-MQNC44ZY', 'L-SS32G0HN', 'L-N9G6AGXW']
    - payment_allocation ['ST002564', 'TX4916733774'] 1717614 CNY evidence=['DOC-ZWD9FS8F', 'L-EGKQTATZ']
    - payment_allocation ['ST002546', 'TX7219962447'] 80460000 INR evidence=['DOC-DGTF752N', 'L-M607FC2Q']
    - fx_conversion ['ST002564', 'TX4916733774'] 16663 INR evidence=['DOC-ZWD9FS8F', 'FX:2025-12-31:CNY:INR', 'L-HEN2F32M', 'L-PASK8PB6']

Result (open training mode): terminated=True truncated=False reward=0.942; TP=4 FP=0 FN=0 P=1.00 R=1.00 F1=1.00
