import csv
import sys

from pydriller import Repository

import repository_scan


def start(repo_link):
    filename = 'generated_dataset.csv'

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', "#CodeChurnInFile",
                         "MaxSizeCodeChurn", "AvgSizeCodeChurn", "#Commits", "#Contributors",
                         "#MinorContributors", "#ContibutorExperience", "#Hunks", "#LinesAdded", "MaxLinesAdded",
                         "AvgLinesAdded", "#LinesRemoved", "MaxLinesRemoved", "AvgLinesRemoved"])

        try:
            print("\n\nRepo link: ", repo_link)

            repository = Repository(repo_link)
            for commit in repository.traverse_commits():
                last_commit = repository.git.get_head()
                last_hash = last_commit.hash

                commit_generator = repository.git.get_list_commits()
                array_commit = list(commit_generator)
                first = array_commit[0]
                first_hash = first.hash
                break

            if last_hash is None or first_hash is None:
                print("\nError during commit search\n")
            else:
                print("First hash: ", first_hash)
                print("Last hash: ", last_hash)
                repository_scan.metric_calculation_and_writing(first_hash, last_hash, repo_link, writer, -1)

        except Exception as e:
            sys.stderr.write(str(e))
