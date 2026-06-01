import generate_inputs as genin

REGION = "Magallanes"
YEAR = 2023
BASE = f"../R/intermediate_data/{REGION}"

for tb in ["stb", "mtb", "stbf", "mtbf"]:
    genin.write_instance_from_csv(
        indir=BASE,
        outdir=BASE,
        tie_breaker=tb,
        yr=YEAR
    )
    print(f"Built {REGION} {YEAR} with tie breaker {tb}")