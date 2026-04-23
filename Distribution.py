import os
import sys
from collections import defaultdict
import matplotlib.pyplot as plt


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".JPG", ".JPEG"}


def count_images_per_class(root_dir: str) -> dict[str, int]:

    if not os.path.isdir(root_dir):
        print(f"Error: '{root_dir}' is not a valid directory.")
        sys.exit(1)

    class_counts = defaultdict(int)

    for current_root, _, files in os.walk(root_dir):
        count = sum(
            1
            for file_name in files
            if os.path.splitext(file_name)[1] in IMAGE_EXTENSIONS
        )

        if count > 0:
            class_name = os.path.basename(current_root)
            class_counts[class_name] += count

    if not class_counts:
        print(f"No image subdirectories found in '{root_dir}'.")
        sys.exit(1)

    return dict(class_counts)


def group_by_plant(class_counts):

    grouped = defaultdict(dict)

    for class_name, count in class_counts.items():
        parts = class_name.split("_", 1)
        plant = parts[0]
        disease = parts[1] if len(parts) > 1 else class_name
        grouped[plant][disease] = count

    return grouped


def plot_distribution(class_counts):

    labels = list(class_counts.keys())
    values = list(class_counts.values())

    cmap = plt.cm.tab20 if len(labels) > 8 else plt.cm.Set2
    colors = cmap.colors[: len(labels)]
    title = "Overall class distribution"

    fig_pie, ax_pie = plt.subplots(figsize=(9, 7))
    fig_pie.canvas.manager.set_window_title("Dataset - Pie Chart")
    fig_pie.suptitle(title, fontsize=14)

    wedges, _, autotexts = ax_pie.pie(
        values,
        labels=None,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        pctdistance=0.75,
    )

    for autotext in autotexts:
        autotext.set_fontsize(9)

    ax_pie.legend(
        wedges,
        labels,
        title="Classes",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.2),
        fontsize=8,
        ncol=2,
    )

    fig_pie.tight_layout()

    fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
    fig_bar.canvas.manager.set_window_title("Dataset - Bar Chart")
    fig_bar.suptitle(title, fontsize=14)

    bars = ax_bar.bar(
        labels,
        values,
        color=colors,
        edgecolor="white",
        width=0.6
    )

    ax_bar.set_xlabel("Disease class", fontsize=10)
    ax_bar.set_ylabel("Number of images", fontsize=10)
    ax_bar.set_xticks(range(len(labels)))
    ax_bar.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax_bar.set_ylim(0, max(values) * 1.15)

    for bar, val in zip(bars, values):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            str(val),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig_bar.tight_layout()


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python Distribution.py <directory>")
        sys.exit(1)

    root_dir = sys.argv[1].rstrip("/").rstrip("\\")

    print(f"Scanning: {root_dir}\n")
    class_counts = count_images_per_class(root_dir)
    grouped = group_by_plant(class_counts)

    for plant, diseases in sorted(grouped.items()):

        print(f"{plant}:")

        for disease, count in sorted(diseases.items()):
            print(f"{disease}: {count} images")

        print(f"Total: {sum(diseases.values())} images\n")

    print(f"Overall Total: {sum(class_counts.values())} images\n")

    plot_distribution(class_counts)

    plt.show()
