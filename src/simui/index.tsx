// Vendored from https://github.com/joyhughes/simple-reactive-ui (@openclaw/sim-ui), which is marked
// private and points `main` at raw TSX, so it cannot be installed as a dependency. Kept close to
// upstream to stay syncable; the only local change is the optional `viewportFill` prop below.

import { useLayoutEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import './styles.css'

type NumericKey<T> = {
  [K in keyof T]: T[K] extends number ? K : never
}[keyof T]

export type NumericControlConfig<T extends object> = {
  key: NumericKey<T>
  label: string
  min: number
  max: number
  step: number
  suffix?: string
}

type AppFrameProps = {
  title: string
  subtitle?: string
  actions?: ReactNode
  controls: ReactNode
  viewport: ReactNode
  stats?: ReactNode
  /**
   * Upstream sizes the viewport as a square, for simulations that render one. An image editor wants the
   * frame to fill its grid cell instead and let the content letterbox itself, which is what this opts into.
   */
  viewportFill?: boolean
  className?: string
  controlsLabel?: string
  viewportLabel?: string
}

export function AppFrame({
  title,
  subtitle,
  actions,
  controls,
  viewport,
  stats,
  viewportFill = false,
  className,
  controlsLabel = 'Controls',
  viewportLabel = 'Viewport',
}: AppFrameProps) {
  const stageRef = useRef<HTMLElement | null>(null)
  const statsRef = useRef<HTMLDivElement | null>(null)
  const [viewportSize, setViewportSize] = useState<number | null>(null)

  useLayoutEffect(() => {
    const stage = stageRef.current

    if (!stage || viewportFill) {
      return
    }

    const measure = () => {
      const statsHeight = statsRef.current?.getBoundingClientRect().height ?? 0
      const stageStyle = getComputedStyle(stage)
      const rowGap = Number.parseFloat(stageStyle.rowGap || '0') || 0
      const availableHeight = stage.clientHeight - statsHeight - (statsHeight > 0 ? rowGap : 0)
      const nextSize = Math.max(1, Math.floor(Math.min(stage.clientWidth, availableHeight)))
      setViewportSize(nextSize)
    }

    const resizeObserver = new ResizeObserver(measure)
    resizeObserver.observe(stage)

    if (statsRef.current) {
      resizeObserver.observe(statsRef.current)
    }

    measure()

    return () => resizeObserver.disconnect()
  }, [stats, viewportFill])

  return (
    <main className={['sim-ui-shell', className].filter(Boolean).join(' ')}>
      <header className="sim-ui-topbar">
        <div className="sim-ui-brand">
          <h1>{title}</h1>
          {subtitle && <p>{subtitle}</p>}
        </div>
        {actions && <div className="sim-ui-actions">{actions}</div>}
      </header>
      <section className="sim-ui-workbench" aria-label={viewportLabel}>
        <section className={['sim-ui-stage', stats ? 'has-stats' : undefined].filter(Boolean).join(' ')} ref={stageRef}>
          <div
            className={['sim-ui-viewport-frame', viewportFill ? 'is-fill' : undefined].filter(Boolean).join(' ')}
            style={!viewportFill && viewportSize ? { width: viewportSize, height: viewportSize } : undefined}
          >
            {viewport}
          </div>
          {stats && (
            <div className="sim-ui-stats-rail" ref={statsRef}>
              {stats}
            </div>
          )}
        </section>
        <ControlDock label={controlsLabel}>{controls}</ControlDock>
      </section>
    </main>
  )
}

export function ControlDock({ children, label = 'Controls' }: { children: ReactNode; label?: string }) {
  return (
    <aside className="sim-ui-control-dock" aria-label={label}>
      <div className="sim-ui-control-tiles">{children}</div>
    </aside>
  )
}

export function ControlGroup({
  title,
  children,
  defaultOpen = true,
}: {
  title: string
  children: ReactNode
  defaultOpen?: boolean
}) {
  return (
    <details className="sim-ui-control-group" open={defaultOpen}>
      <summary>{title}</summary>
      <div className="sim-ui-control-group-body">{children}</div>
    </details>
  )
}

export function SegmentedControl<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: T
  options: Array<{ value: T; label: string }>
  onChange: (value: T) => void
}) {
  return (
    <div className="sim-ui-segmented" aria-label={label}>
      {options.map((option) => (
        <button
          className={option.value === value ? 'active' : ''}
          key={option.value}
          type="button"
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  )
}

export function NumericControl<T extends object>({
  item,
  values,
  disabled,
  onChange,
}: {
  item: NumericControlConfig<T>
  values: T
  disabled?: boolean
  onChange: (key: NumericKey<T>, value: number) => void
}) {
  const value = Number(values[item.key])

  return (
    <label className="sim-ui-control">
      <span>
        {item.label}
        <strong>
          {formatNumericValue(value, item.step)}
          {item.suffix ?? ''}
        </strong>
      </span>
      <input
        type="range"
        min={item.min}
        max={item.max}
        step={item.step}
        disabled={disabled}
        value={value}
        onChange={(event) => onChange(item.key, Number(event.target.value))}
      />
    </label>
  )
}

export function SelectControl<T extends string>({
  label,
  value,
  options,
  disabled,
  onChange,
}: {
  label: string
  value: T
  options: Array<{ value: T; label: string }>
  disabled?: boolean
  onChange: (value: T) => void
}) {
  return (
    <label className="sim-ui-select-control">
      <span>{label}</span>
      <select disabled={disabled} value={value} onChange={(event) => onChange(event.target.value as T)}>
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  )
}

export function ToggleControl({
  label,
  checked,
  onChange,
}: {
  label: string
  checked: boolean
  onChange: (checked: boolean) => void
}) {
  return (
    <label className="sim-ui-toggle">
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <span>{label}</span>
    </label>
  )
}

export function StatGrid({ children }: { children: ReactNode }) {
  return <div className="sim-ui-stat-grid">{children}</div>
}

export function StatItem({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="sim-ui-stat-item">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

// Upstream exports this helper alongside the components; kept as-is so the vendored copy stays syncable.
// eslint-disable-next-line react-refresh/only-export-components
export function formatNumericValue(value: number, step: number) {
  return value.toFixed(step < 0.1 ? 2 : step < 1 ? 1 : 0)
}
