with open("bin/start_template", "r", encoding='utf-8') as f:
    template = f.read()

with open("queue_all.sh", "w", encoding="utf-8") as f:
    f.write("#!/bin/bash\n")
for dataset in ["DiatomSizeReduction", "MiddlePhalanxTW", "MiddlePhalanxOutlineCorrect",
                "ProximalPhalanxOutlineAgeGroup", "Meat", "MiddlePhalanxOutlineAgeGroup", "OliveOil",
                "ProximalPhalanxTW"]:
    formatted_template = template.format(dataset=dataset, cpus=10,
                                         methods="1_vs_all asymmetric_1_vs_1 symmetric_1_vs_1 "
                                                 "combined_symmetric_1_vs_1",
                                         configs="1.ini 2.ini 3.ini 4.ini 5.ini 6.ini 7.ini 8.ini",
                                         test_length_fractions="1 0.8 0.6 0.4")
    with open(f"bin/start_{dataset}.sh", "w", encoding='utf-8') as f:
        f.write(formatted_template)
    with open("queue_all.sh", "a", encoding="utf-8") as f:
        f.write(f"sbatch bin/start_{dataset}.sh\n")
