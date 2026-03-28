"""
extract_features.py
Extracts features directly from ClinVar VCF INFO fields.
No VEP required — uses MC, AF_ESP, AF_EXAC, AF_TGP, GENEINFO fields.
"""

import gzip
import re
import numpy as np
import pandas as pd
from pathlib import Path

# Molecular consequence severity mapping
MC_SEVERITY = {
    "stop_gained":             8,
    "frameshift_variant":      8,
    "splice_acceptor_variant": 7,
    "splice_donor_variant":    7,
    "stop_lost":               6,
    "start_lost":              6,
    "inframe_insertion":       4,
    "inframe_deletion":        4,
    "missense_variant":        4,
    "splice_region_variant":   3,
    "synonymous_variant":      1,
    "intron_variant":          0,
    "intergenic_variant":      0,
    "non_coding_transcript":   0,
}

def safe_float(val, default=np.nan):
    try:
        return float(val) if val and val != "." else default
    except ValueError:
        return default

def parse_info(info_str):
    info = {}
    for item in info_str.split(";"):
        if "=" in item:
            k, v = item.split("=", 1)
            info[k] = v
        else:
            info[item] = True
    return info

def parse_vcf(vcf_path, label):
    rows = []
    opener = gzip.open if vcf_path.endswith(".gz") else open

    with opener(vcf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue

            cols  = line.rstrip("\n").split("\t")
            chrom = cols[0]
            pos   = cols[1]
            ref   = cols[3]
            alt   = cols[4]
            info  = parse_info(cols[7])

            # variant type
            var_type = "SNV" if (len(ref) == 1 and len(alt) == 1) else "INDEL"
            is_indel = 1 if var_type == "INDEL" else 0

            # molecular consequence — take first/most severe
            mc_raw  = info.get("MC", "")
            mc_term = ""
            mc_sev  = 0
            if mc_raw:
                # format: SO:0001587|stop_gained
                parts = mc_raw.split(",")[0]
                if "|" in parts:
                    mc_term = parts.split("|")[1]
                    mc_sev  = MC_SEVERITY.get(mc_term, 0)

            # allele frequencies
            af_esp  = safe_float(info.get("AF_ESP"))
            af_exac = safe_float(info.get("AF_EXAC"))
            af_tgp  = safe_float(info.get("AF_TGP"))

            # take max non-null AF as proxy for population frequency
            afs = [x for x in [af_esp, af_exac, af_tgp] if not np.isnan(x)]
            max_af = max(afs) if afs else np.nan

            # log transform
            af_esp_log  = np.log10(af_esp  + 1e-8) if not np.isnan(af_esp)  else np.nan
            af_exac_log = np.log10(af_exac + 1e-8) if not np.isnan(af_exac) else np.nan
            af_tgp_log  = np.log10(af_tgp  + 1e-8) if not np.isnan(af_tgp)  else np.nan
            max_af_log  = np.log10(max_af  + 1e-8) if not np.isnan(max_af)  else np.nan

            # gene info
            gene = info.get("GENEINFO", "").split(":")[0]

            # variant origin
            origin = info.get("ORIGIN", "0")
            is_germline = 1 if origin == "1" else 0

            # number of submissions (proxy for evidence strength)
            n_submitters = safe_float(info.get("NS"), default=0)

            rows.append({
                "chrom":          chrom,
                "pos":            int(pos),
                "ref":            ref,
                "alt":            alt,
                "gene":           gene,
                "var_type":       var_type,
                "is_indel":       is_indel,
                "mc_term":        mc_term,
                "mc_severity":    mc_sev,
                "af_esp":         af_esp,
                "af_exac":        af_exac,
                "af_tgp":         af_tgp,
                "max_af":         max_af,
                "af_esp_log":     af_esp_log,
                "af_exac_log":    af_exac_log,
                "af_tgp_log":     af_tgp_log,
                "max_af_log":     max_af_log,
                "is_germline":    is_germline,
                "n_submitters":   n_submitters,
                "label":          label,
            })

    return pd.DataFrame(rows)

def main():
    path_vcf = str(snakemake.input.pathogenic)  # noqa
    ben_vcf  = str(snakemake.input.benign)      # noqa
    out_path = str(snakemake.output.tsv)        # noqa

    print("[extract_features] parsing pathogenic variants...")
    df_path = parse_vcf(path_vcf, label=1)
    print(f"  {len(df_path)} variants")

    print("[extract_features] parsing benign variants...")
    df_ben = parse_vcf(ben_vcf, label=0)
    print(f"  {len(df_ben)} variants")

    df = pd.concat([df_path, df_ben], ignore_index=True)
    print(f"[extract_features] total: {len(df)} variants")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"[extract_features] wrote {out_path}")

if __name__ == "__main__":
    main()