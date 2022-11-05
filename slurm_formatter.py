with open("start_template", "r", encoding='utf-8') as f:
    template = f.read()

with open("queue_all.sh", "w", encoding="utf-8") as f:
    f.write("#!/bin/bash\n")
for dataset in ["DiatomSizeReduction",
                # "MiddlePhalanxTW",
                # "MiddlePhalanxOutlineCorrect",
                # "ProximalPhalanxOutlineAgeGroup",
                "Meat",
                # "MiddlePhalanxOutlineAgeGroup",
                "OliveOil",
                # "ProximalPhalanxTW"
                ]:
    formatted_template = template.format(dataset=dataset, cpus=10,
                                         methods="1_vs_all asymmetric_1_vs_1 symmetric_1_vs_1 "
                                                 "combined_symmetric_1_vs_1",
                                         configs="small0.ini small1.ini small2.ini small3.ini small4.ini small5.ini small6.ini small7.ini small8.ini small9.ini small10.ini small11.ini",
                                         test_length_fractions="1 0.8 0.6 0.4")
    with open(f"bin/start_{dataset}.sh", "w", encoding='utf-8') as f:
        f.write(formatted_template)
    with open("queue_all.sh", "a", encoding="utf-8") as f:
        f.write(f"sbatch bin/start_{dataset}.sh\n")
