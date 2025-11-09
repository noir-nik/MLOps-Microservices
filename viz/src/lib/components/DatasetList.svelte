<script lang="ts">
	interface Props {
		datasets: any[];
		API_URL: string;
	}

	let { datasets, API_URL }: Props = $props();

	let hoveredId = $state<number | null>(null);

	function downloadDataset(id: number) {
		window.open(`${API_URL}/datasets/${id}/download`, '_blank');
	}
</script>

<div class="bg-white shadow rounded-lg">
	<div class="px-6 py-4 border-gray-200 border-b">
		<h3 class="font-semibold text-gray-900 text-lg">Datasets ({datasets.length})</h3>
	</div>

	<div class="divide-y divide-gray-200 max-h-96 overflow-y-auto">
		{#if datasets.length === 0}
			<div class="px-6 py-8 text-gray-500 text-center">No datasets yet. Use "Collect & Store" to create one.</div>
		{:else}
			{#each datasets as dataset}
				<!-- svelte-ignore a11y_no_static_element_interactions -->
				<div
					class="relative hover:bg-gray-50 px-6 py-3 transition-colors"
					onmouseenter={() => (hoveredId = dataset.id)}
					onmouseleave={() => (hoveredId = null)}
				>
					<div class="flex justify-between items-start">
						<div class="flex-1 min-w-0">
							<p class="font-medium text-gray-900 text-sm truncate">{dataset.name}</p>
							<p class="mt-1 text-gray-500 text-xs">
								{dataset.rows} rows × {dataset.columns} cols
							</p>
							<p class="mt-1 text-gray-400 text-xs">
								{new Date(dataset.created_at).toLocaleString()}
							</p>
						</div>

						<!-- {#if hoveredId === dataset.id} -->
						{#if true}
							<a
								href={`${API_URL}/datasets/${dataset.id}/download`}
								class="bg-blue-100 hover:bg-blue-200 ml-2 px-2 py-1 rounded text-blue-700 text-xs transition-colors"
							>
								Download
							</a>
						{/if}
					</div>

					<!-- {#if hoveredId === dataset.id && dataset.meta} -->
					{#if true && dataset.meta}
						<div class="bg-gray-50 mt-2 p-2 rounded text-xs">
							<p class="text-gray-600">
								<strong>Metadata:</strong>
								{JSON.stringify(dataset.meta, null, 2)}
							</p>
						</div>
					{/if}
				</div>
			{/each}
		{/if}
	</div>
</div>
