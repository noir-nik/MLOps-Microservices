<script lang="ts">
	interface Props {
		models: any[];
		API_URL: string;
		openReport: (model: any) => void;
	}

	let { models, API_URL, openReport }: Props = $props();

	let hoveredId = $state<number | null>(null);

	function downloadModel(id: number) {
		window.open(`${API_URL}/models/${id}/download`, '_blank');
	}
</script>

<div class="bg-white shadow rounded-lg">
	<div class="px-6 py-4 border-gray-200 border-b">
		<h3 class="font-semibold text-gray-900 text-lg">Models ({models.length})</h3>
	</div>

	<div class="divide-y divide-gray-200 max-h-96 overflow-y-auto">
		{#if models.length === 0}
			<div class="px-6 py-8 text-gray-500 text-center">No models yet. Use "Train Model" to create one.</div>
		{:else}
			{#each models as model}
				<!-- svelte-ignore a11y_no_static_element_interactions -->
				<div
					class="relative hover:bg-gray-50 px-6 py-3 transition-colors"
					onmouseenter={() => (hoveredId = model.id)}
					onmouseleave={() => (hoveredId = null)}
				>
					<div class="flex justify-between items-start">
						<div class="flex-1 min-w-0">
							<p class="font-medium text-gray-900 text-sm truncate">{model.name}</p>
							<p class="mt-1 text-gray-600 text-xs">{model.model_type}</p>
							{#if model.accuracy}
								<p class="mt-1 font-medium text-green-600 text-xs">
									Accuracy: {(model.accuracy * 100).toFixed(2)}%
								</p>
							{/if}
							<p class="mt-1 text-gray-400 text-xs">
								{new Date(model.created_at).toLocaleString()}
							</p>
						</div>

						<!-- {#if hoveredId === model.id} -->
						{#if true}
							<div class="flex flex-col gap-1">
								<button
									onclick={() => openReport(model)}
									class="bg-purple-100 hover:bg-purple-200 px-2 py-1 rounded text-purple-700 text-xs transition-colors"
								>
									Report
								</button>
								<a
									href={`${API_URL}/models/${model.id}/download`}
									class="bg-blue-100 hover:bg-blue-200 px-2 py-1 rounded text-blue-700 text-xs transition-colors"
								>
									Download
								</a>
							</div>
						{/if}
					</div>

					{#if true && model.meta}
						<div class="bg-gray-50 mt-2 p-2 rounded max-h-32 overflow-y-auto text-xs">
							<p class="text-gray-600">
								<strong>Hyperparameters:</strong>
								{JSON.stringify(model.meta.hyperparameters, null, 2)}
							</p>
						</div>
					{/if}
				</div>
			{/each}
		{/if}
	</div>
</div>
