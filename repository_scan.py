import pandas as pd
from pydriller import Repository


def main():
    cve = pd.read_csv("../data/bigvul.csv")

    cve = cve.head(100)

    for x in range(len(cve)):
        commit_link = cve.iloc[0]['ref_link']
        print(commit_link)

        # Removal of user and commit hash from link
        repo_link = commit_link[:int(len(commit_link) - 48)]
        print(repo_link)

        for commit in Repository(str(repo_link)).traverse_commits():
            print(commit.hash)
            print(commit.msg)
            print(commit.author.name)

            for file in commit.modified_files:
                print(file.filename, ' has changed')

            print("\n\n\n\n\nEND OF REPOSITORY COMMITS\n\n\n\n\n")


if __name__ == "__main__":
    main()