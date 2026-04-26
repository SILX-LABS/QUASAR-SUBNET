import { redirect } from "next/navigation";

export const dynamic = "force-static";

export default function HomePage() {
  redirect("/index.html");
}
