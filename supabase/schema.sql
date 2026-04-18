-- MarketPulse core schema (run in Supabase SQL editor or via migration CLI)
-- Requires pgcrypto for gen_random_uuid (enabled by default on Supabase)

create table if not exists public.profiles (
  id uuid references auth.users on delete cascade primary key,
  email text,
  tier text not null default 'free' check (tier in ('free', 'pro', 'admin')),
  forecasts_today int not null default 0,
  forecasts_today_reset date,
  chat_messages_today int not null default 0,
  chat_reset date,
  created_at timestamptz not null default now(),
  last_active timestamptz default now()
);

create table if not exists public.watchlists (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references public.profiles(id) on delete cascade,
  symbol text not null,
  added_at timestamptz not null default now(),
  unique (user_id, symbol)
);

create table if not exists public.forecasts (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references public.profiles(id) on delete cascade,
  symbol text not null,
  horizon int not null,
  last_close double precision,
  predicted_prices double precision[] not null,
  predicted_dates text[] not null,
  model_mae double precision,
  actual_close double precision,
  accuracy_pct double precision,
  created_at timestamptz not null default now()
);

create table if not exists public.chat_sessions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references public.profiles(id) on delete cascade,
  messages jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.alerts (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references public.profiles(id) on delete cascade,
  symbol text not null,
  condition text check (condition in ('above', 'below', 'move_pct')),
  threshold double precision not null,
  active boolean not null default true,
  triggered_at timestamptz,
  created_at timestamptz not null default now()
);

create table if not exists public.portfolios (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references public.profiles(id) on delete cascade,
  symbol text not null,
  shares double precision,
  avg_cost double precision,
  added_at timestamptz not null default now()
);

create table if not exists public.admin_settings (
  key text primary key,
  value jsonb not null,
  updated_by uuid references public.profiles(id),
  updated_at timestamptz not null default now()
);

insert into public.admin_settings (key, value) values
  ('rate_limits', '{"public": 3, "free": 15, "pro": 999}'),
  ('max_horizon', '{"public": 7, "free": 14, "pro": 30}'),
  ('chatbot_system_prompt', '"You are a financial analyst assistant for MarketPulse."'),
  ('kill_switch', '{"enabled": false, "reason": ""}'),
  ('disabled_tickers', '[]')
on conflict (key) do nothing;

create table if not exists public.flagged_conversations (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references public.profiles(id),
  session_id uuid references public.chat_sessions(id),
  reason text,
  reviewed boolean not null default false,
  created_at timestamptz not null default now()
);

create table if not exists public.announcements (
  id uuid primary key default gen_random_uuid(),
  message text not null,
  active boolean not null default true,
  created_by uuid references public.profiles(id),
  created_at timestamptz not null default now(),
  expires_at timestamptz
);

alter table public.profiles enable row level security;
create policy "profiles_select_own" on public.profiles for select using (auth.uid() = id);
create policy "profiles_update_own" on public.profiles for update using (auth.uid() = id);

alter table public.watchlists enable row level security;
create policy "watchlists_own" on public.watchlists for all using (auth.uid() = user_id);

alter table public.forecasts enable row level security;
create policy "forecasts_own" on public.forecasts for all using (auth.uid() = user_id);

alter table public.chat_sessions enable row level security;
create policy "chat_sessions_own" on public.chat_sessions for all using (auth.uid() = user_id);

-- Service role bypasses RLS; anon/authenticated clients use policies above.

create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.profiles (id, email)
  values (new.id, new.email)
  on conflict (id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();
