import { FEATURE_NETWORKS, type FeatureNetworkId } from '../ml/featureModel';

interface FeatureNetworkPickerProps {
  value: FeatureNetworkId;
  onChange: (id: FeatureNetworkId) => void;
  isLoading: boolean;
  disabled: boolean;
}

export function FeatureNetworkPicker({ value, onChange, isLoading, disabled }: FeatureNetworkPickerProps) {
  const selected = FEATURE_NETWORKS.find((network) => network.id === value);

  return (
    <div className="preset-panel">
      <label
        className="field-row"
        title="Which pretrained network's features drive the effect. Style transfer matches the statistics of this network's intermediate layers, so the choice of network shapes the result more than any slider does."
      >
        <span>Feature network</span>
        <select
          value={value}
          onChange={(e) => onChange(e.target.value as FeatureNetworkId)}
          disabled={disabled || isLoading}
        >
          {FEATURE_NETWORKS.map((network) => (
            <option key={network.id} value={network.id}>
              {network.label} ({network.downloadLabel})
            </option>
          ))}
        </select>
      </label>
      <p className="field-hint">
        {isLoading ? `Downloading ${selected?.label} (${selected?.downloadLabel})…` : selected?.description}
      </p>
    </div>
  );
}
